// test.cpp — Matrix Multiplication Benchmark
// Compares: Naive, AVX2+FMA, Vulkan compute shader, pthread, pthread+AVX2

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <immintrin.h>   // AVX2 / FMA
#include <pthread.h>
#include <thread>            // std::thread::hardware_concurrency
#include <vulkan/vulkan.h>

static const uint32_t N = 8192; // Matrix dimension NxN
static const int deviceNo = 0; // Adjust if you have multiple GPUs, otherwise leave it at 0

// ============================================================
// Utility
// ============================================================

static void fill_random(std::vector<float>& v) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

// ============================================================
// Function 1: Naive triple-loop matrix multiplication
// ============================================================

static void matmul_naive(const float* A, const float* B, float* C, uint32_t n) {
    std::fill(C, C + n * n, 0.0f);
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t k = 0; k < n; k++) {
            float aik = A[i * n + k];
            for (uint32_t j = 0; j < n; j++) {
                C[i * n + j] += aik * B[k * n + j];
            }
        }
    }
}

// ============================================================
// Function 2: AVX2 + FMA accelerated matrix multiplication
//   Strategy: transpose B for cache-friendly dot products,
//   then use 256-bit FMA to compute 8 products at once.
// ============================================================

static void matmul_avx2(const float* A, const float* B, float* C, uint32_t n) {
    std::vector<float> BT(n * n);
    for (uint32_t i = 0; i < n; i++)
        for (uint32_t j = 0; j < n; j++)
            BT[j * n + i] = B[i * n + j];

    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < n; j++) {
            __m256 vsum = _mm256_setzero_ps();
            const float* rowA  = A  + i * n;
            const float* rowBT = BT.data() + j * n;
            uint32_t k = 0;
            for (; k + 8 <= n; k += 8) {
                __m256 va = _mm256_loadu_ps(rowA  + k);
                __m256 vb = _mm256_loadu_ps(rowBT + k);
                vsum = _mm256_fmadd_ps(va, vb, vsum);
            }
            // Horizontal sum of vsum (256-bit -> scalar)
            __m128 lo  = _mm256_castps256_ps128(vsum);
            __m128 hi  = _mm256_extractf128_ps(vsum, 1);
            __m128 s   = _mm_add_ps(lo, hi);
            s = _mm_hadd_ps(s, s);
            s = _mm_hadd_ps(s, s);
            float total = _mm_cvtss_f32(s);
            // Scalar tail
            for (; k < n; k++)
                total += rowA[k] * rowBT[k];
            C[i * n + j] = total;
        }
    }
}

// ============================================================
// Function 4: pthread parallel matrix multiplication
//   Strategy: partition output rows evenly across threads.
//   Each thread computes its slice independently — no locks needed
//   because every thread writes to a disjoint region of C.
// ============================================================

static int NTHREADS = 0; // 0 = auto-detect via hardware_concurrency

struct PthreadArgs {
    const float* A;
    const float* B;   // original B (pthread naive worker)
    const float* BT;  // transposed B (pthread+AVX2 worker), may be null
    float*       C;
    int          n;
    int          row_start; // inclusive
    int          row_end;   // exclusive
};

static void* pthread_worker(void* arg) {
    auto* a = static_cast<PthreadArgs*>(arg);
    const float* A = a->A;
    const float* B = a->B;
    float*       C = a->C;
    int          n = a->n;

    for (int i = a->row_start; i < a->row_end; i++) {
        for (int k = 0; k < n; k++) {
            float aik = A[i * n + k];
            for (int j = 0; j < n; j++)
                C[i * n + j] += aik * B[k * n + j];
        }
    }
    return nullptr;
}

static void matmul_pthread(const float* A, const float* B, float* C, int n) {
    std::fill(C, C + n * n, 0.0f);

    int nthreads = NTHREADS > 0
                   ? NTHREADS
                   : (int)std::thread::hardware_concurrency();
    if (nthreads < 1) nthreads = 4;
    if (nthreads > n) nthreads = n; // no point having more threads than rows

    std::vector<pthread_t>    threads(nthreads);
    std::vector<PthreadArgs>  args(nthreads);

    int rows_per_thread = n / nthreads;
    int remainder       = n % nthreads;
    int row = 0;

    for (int t = 0; t < nthreads; t++) {
        int rows        = rows_per_thread + (t < remainder ? 1 : 0);
        args[t]         = {A, B, nullptr, C, n, row, row + rows};
        row            += rows;
        pthread_create(&threads[t], nullptr, pthread_worker, &args[t]);
    }

    for (int t = 0; t < nthreads; t++)
        pthread_join(threads[t], nullptr);
}

// ============================================================
// Function 5: pthread + AVX2 matrix multiplication
//   Each thread runs the same AVX2+FMA dot-product kernel on its
//   assigned row slice. B is transposed once on the main thread
//   before spawning so all workers share a single read-only BT.
// ============================================================

static void* pthread_avx2_worker(void* arg) {
    auto* a        = static_cast<PthreadArgs*>(arg);
    const float* A  = a->A;
    const float* BT = a->BT; // transposed B
    float*       C  = a->C;
    int          n  = a->n;

    for (int i = a->row_start; i < a->row_end; i++) {
        for (int j = 0; j < n; j++) {
            __m256 vsum        = _mm256_setzero_ps();
            const float* rowA  = A  + i * n;
            const float* rowBT = BT + j * n;
            int k = 0;
            for (; k + 8 <= n; k += 8) {
                __m256 va = _mm256_loadu_ps(rowA  + k);
                __m256 vb = _mm256_loadu_ps(rowBT + k);
                vsum = _mm256_fmadd_ps(va, vb, vsum);
            }
            // Horizontal sum
            __m128 lo = _mm256_castps256_ps128(vsum);
            __m128 hi = _mm256_extractf128_ps(vsum, 1);
            __m128 s  = _mm_add_ps(lo, hi);
            s = _mm_hadd_ps(s, s);
            s = _mm_hadd_ps(s, s);
            float total = _mm_cvtss_f32(s);
            for (; k < n; k++)
                total += rowA[k] * rowBT[k];
            C[i * n + j] = total;
        }
    }
    return nullptr;
}

static void matmul_pthread_avx2(const float* A, const float* B, float* C, int n) {
    // Transpose B once on the calling thread; workers only read it.
    std::vector<float> BT(n * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            BT[j * n + i] = B[i * n + j];

    int nthreads = NTHREADS > 0
                   ? NTHREADS
                   : (int)std::thread::hardware_concurrency();
    if (nthreads < 1) nthreads = 4;
    if (nthreads > n) nthreads = n;

    std::vector<pthread_t>   threads(nthreads);
    std::vector<PthreadArgs> args(nthreads);

    int rows_per_thread = n / nthreads;
    int remainder       = n % nthreads;
    int row = 0;

    for (int t = 0; t < nthreads; t++) {
        int rows  = rows_per_thread + (t < remainder ? 1 : 0);
        args[t]   = {A, nullptr, BT.data(), C, n, row, row + rows};
        row      += rows;
        pthread_create(&threads[t], nullptr, pthread_avx2_worker, &args[t]);
    }

    for (int t = 0; t < nthreads; t++)
        pthread_join(threads[t], nullptr);
}

// ============================================================
// Function 3: Vulkan compute shader matrix multiplication
// ============================================================

static std::vector<uint32_t> loadSPIRV(const std::string& path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Cannot open SPIR-V: " + path);
    size_t sz = (size_t)f.tellg();
    if (sz % 4 != 0) throw std::runtime_error("SPIR-V size not multiple of 4");
    std::vector<uint32_t> code(sz / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(code.data()), sz);
    return code;
}

static uint32_t findComputeQueue(VkPhysicalDevice pd) {
    uint32_t cnt = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &cnt, nullptr);
    std::vector<VkQueueFamilyProperties> props(cnt);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &cnt, props.data());
    for (uint32_t i = 0; i < cnt; i++)
        if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) return i;
    throw std::runtime_error("No compute queue family");
}

static uint32_t findMemType(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    throw std::runtime_error("No suitable memory type");
}

static void makeBuffer(VkDevice dev, VkPhysicalDevice pd, VkDeviceSize size,
                       VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps,
                       VkBuffer& buf, VkDeviceMemory& mem) {
    VkBufferCreateInfo bci{};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = size;
    bci.usage       = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(dev, &bci, nullptr, &buf) != VK_SUCCESS)
        throw std::runtime_error("vkCreateBuffer failed");

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, buf, &mr);

    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = mr.size;
    ai.memoryTypeIndex = findMemType(pd, mr.memoryTypeBits, memProps);
    if (vkAllocateMemory(dev, &ai, nullptr, &mem) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateMemory failed");
    vkBindBufferMemory(dev, buf, mem, 0);
}

class VulkanMatMul {
public:
    VulkanMatMul(uint32_t n, const std::string& spirvPath) : n_(n) {
        // Instance
        VkApplicationInfo app{};
        app.sType      = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo ici{};
        ici.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &app;
        if (vkCreateInstance(&ici, nullptr, &instance_) != VK_SUCCESS)
            throw std::runtime_error("vkCreateInstance failed");

        // Physical device
        uint32_t cnt = 0;
        vkEnumeratePhysicalDevices(instance_, &cnt, nullptr);
        if (!cnt) throw std::runtime_error("No Vulkan GPU found");
        std::vector<VkPhysicalDevice> pds(cnt);
        vkEnumeratePhysicalDevices(instance_, &cnt, pds.data());
        pd_ = pds[deviceNo];

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd_, &props);
        std::cout << "  [Vulkan] GPU: " << props.deviceName << "\n";

        // Logical device
        qf_ = findComputeQueue(pd_);
        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = qf_;
        qci.queueCount       = 1;
        qci.pQueuePriorities = &prio;

        VkDeviceCreateInfo dci{};
        dci.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos    = &qci;
        if (vkCreateDevice(pd_, &dci, nullptr, &dev_) != VK_SUCCESS)
            throw std::runtime_error("vkCreateDevice failed");
        vkGetDeviceQueue(dev_, qf_, 0, &queue_);

        // Device-local buffers for compute (A, B: TRANSFER_DST | STORAGE; C: TRANSFER_SRC | STORAGE)
        VkDeviceSize sz = (VkDeviceSize)n_ * n_ * sizeof(float);
        makeBuffer(dev_, pd_, sz,
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                   bufA_, memA_);
        makeBuffer(dev_, pd_, sz,
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                   bufB_, memB_);
        makeBuffer(dev_, pd_, sz,
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                   bufC_, memC_);

        // Staging buffers (HOST_VISIBLE | HOST_COHERENT) for CPU upload/download
        makeBuffer(dev_, pd_, sz,
                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   stagBufA_, stagMemA_);
        makeBuffer(dev_, pd_, sz,
                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   stagBufB_, stagMemB_);
        makeBuffer(dev_, pd_, sz,
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   stagBufC_, stagMemC_);

        // Shader module
        auto spirv = loadSPIRV(spirvPath);
        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spirv.size() * 4;
        smci.pCode    = spirv.data();
        if (vkCreateShaderModule(dev_, &smci, nullptr, &shader_) != VK_SUCCESS)
            throw std::runtime_error("vkCreateShaderModule failed");

        // Descriptor set layout (3 storage buffers)
        VkDescriptorSetLayoutBinding binds[3]{};
        for (int i = 0; i < 3; i++) {
            binds[i].binding         = i;
            binds[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = 3;
        dslci.pBindings    = binds;
        vkCreateDescriptorSetLayout(dev_, &dslci, nullptr, &dsl_);

        // Push constant: uint N
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size       = sizeof(uint32_t);

        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dsl_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        vkCreatePipelineLayout(dev_, &plci, nullptr, &pipeLayout_);

        // Compute pipeline
        VkComputePipelineCreateInfo cpci{};
        cpci.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = shader_;
        cpci.stage.pName  = "main";
        cpci.layout       = pipeLayout_;
        if (vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline_) != VK_SUCCESS)
            throw std::runtime_error("vkCreateComputePipelines failed");

        // Descriptor pool + set
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes    = &ps;
        vkCreateDescriptorPool(dev_, &dpci, nullptr, &descPool_);

        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = descPool_;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts        = &dsl_;
        vkAllocateDescriptorSets(dev_, &dsai, &descSet_);

        // Write descriptors
        VkDescriptorBufferInfo binfos[3] = {
            {bufA_, 0, sz}, {bufB_, 0, sz}, {bufC_, 0, sz}
        };
        VkWriteDescriptorSet writes[3]{};
        for (int i = 0; i < 3; i++) {
            writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet          = descSet_;
            writes[i].dstBinding      = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo     = &binfos[i];
        }
        vkUpdateDescriptorSets(dev_, 3, writes, 0, nullptr);

        // Command pool
        VkCommandPoolCreateInfo cpoolci{};
        cpoolci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpoolci.queueFamilyIndex = qf_;
        cpoolci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        vkCreateCommandPool(dev_, &cpoolci, nullptr, &cmdPool_);

        VkCommandBufferAllocateInfo cbai{};
        cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool        = cmdPool_;
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        vkAllocateCommandBuffers(dev_, &cbai, &cmdBuf_);

        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCreateFence(dev_, &fci, nullptr, &fence_);
    }

    // Upload A and B to staging, DMA to VRAM, dispatch compute, DMA result back, download C.
    // Only this function is timed in main().
    void compute(const float* A, const float* B, float* C) {
        VkDeviceSize sz = (VkDeviceSize)n_ * n_ * sizeof(float);

        // Write A and B into host-visible staging buffers
        auto upload = [&](VkDeviceMemory mem, const float* src) {
            void* ptr;
            vkMapMemory(dev_, mem, 0, sz, 0, &ptr);
            memcpy(ptr, src, sz);
            vkUnmapMemory(dev_, mem);
        };
        upload(stagMemA_, A);
        upload(stagMemB_, B);

        // Record command buffer
        vkResetCommandBuffer(cmdBuf_, 0);
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmdBuf_, &bi);

        // DMA staging -> device-local for A and B
        VkBufferCopy region{0, 0, sz};
        vkCmdCopyBuffer(cmdBuf_, stagBufA_, bufA_, 1, &region);
        vkCmdCopyBuffer(cmdBuf_, stagBufB_, bufB_, 1, &region);

        // Barrier: transfer writes must be visible to the compute shader
        VkMemoryBarrier bar{};
        bar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmdBuf_,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &bar, 0, nullptr, 0, nullptr);

        // Dispatch compute
        vkCmdBindPipeline(cmdBuf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(cmdBuf_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeLayout_, 0, 1, &descSet_, 0, nullptr);
        vkCmdPushConstants(cmdBuf_, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(uint32_t), &n_);
        uint32_t groups = (n_ + 15) / 16;
        vkCmdDispatch(cmdBuf_, groups, groups, 1);

        // Barrier: shader writes to C must be visible to the transfer
        bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmdBuf_,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &bar, 0, nullptr, 0, nullptr);

        // DMA device-local C -> staging
        vkCmdCopyBuffer(cmdBuf_, bufC_, stagBufC_, 1, &region);

        vkEndCommandBuffer(cmdBuf_);

        // Submit and wait
        vkResetFences(dev_, 1, &fence_);
        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cmdBuf_;
        vkQueueSubmit(queue_, 1, &si, fence_);
        vkWaitForFences(dev_, 1, &fence_, VK_TRUE, UINT64_MAX);

        // Download C from staging
        void* ptr;
        vkMapMemory(dev_, stagMemC_, 0, sz, 0, &ptr);
        memcpy(C, ptr, sz);
        vkUnmapMemory(dev_, stagMemC_);
    }

    ~VulkanMatMul() {
        vkDestroyFence(dev_, fence_, nullptr);
        vkDestroyCommandPool(dev_, cmdPool_, nullptr);
        vkDestroyDescriptorPool(dev_, descPool_, nullptr);
        vkDestroyPipeline(dev_, pipeline_, nullptr);
        vkDestroyPipelineLayout(dev_, pipeLayout_, nullptr);
        vkDestroyDescriptorSetLayout(dev_, dsl_, nullptr);
        vkDestroyShaderModule(dev_, shader_, nullptr);
        vkDestroyBuffer(dev_, stagBufC_, nullptr); vkFreeMemory(dev_, stagMemC_, nullptr);
        vkDestroyBuffer(dev_, stagBufB_, nullptr); vkFreeMemory(dev_, stagMemB_, nullptr);
        vkDestroyBuffer(dev_, stagBufA_, nullptr); vkFreeMemory(dev_, stagMemA_, nullptr);
        vkDestroyBuffer(dev_, bufC_, nullptr); vkFreeMemory(dev_, memC_, nullptr);
        vkDestroyBuffer(dev_, bufB_, nullptr); vkFreeMemory(dev_, memB_, nullptr);
        vkDestroyBuffer(dev_, bufA_, nullptr); vkFreeMemory(dev_, memA_, nullptr);
        vkDestroyDevice(dev_, nullptr);
        vkDestroyInstance(instance_, nullptr);
    }

private:
    uint32_t n_;
    VkInstance            instance_  = VK_NULL_HANDLE;
    VkPhysicalDevice      pd_        = VK_NULL_HANDLE;
    VkDevice              dev_       = VK_NULL_HANDLE;
    VkQueue               queue_     = VK_NULL_HANDLE;
    uint32_t              qf_        = 0;
    VkBuffer              bufA_      = VK_NULL_HANDLE;
    VkBuffer              bufB_      = VK_NULL_HANDLE;
    VkBuffer              bufC_      = VK_NULL_HANDLE;
    VkDeviceMemory        memA_      = VK_NULL_HANDLE;
    VkDeviceMemory        memB_      = VK_NULL_HANDLE;
    VkDeviceMemory        memC_      = VK_NULL_HANDLE;
    VkBuffer              stagBufA_  = VK_NULL_HANDLE;
    VkBuffer              stagBufB_  = VK_NULL_HANDLE;
    VkBuffer              stagBufC_  = VK_NULL_HANDLE;
    VkDeviceMemory        stagMemA_  = VK_NULL_HANDLE;
    VkDeviceMemory        stagMemB_  = VK_NULL_HANDLE;
    VkDeviceMemory        stagMemC_  = VK_NULL_HANDLE;
    VkShaderModule        shader_    = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl_       = VK_NULL_HANDLE;
    VkPipelineLayout      pipeLayout_= VK_NULL_HANDLE;
    VkPipeline            pipeline_  = VK_NULL_HANDLE;
    VkDescriptorPool      descPool_  = VK_NULL_HANDLE;
    VkDescriptorSet       descSet_   = VK_NULL_HANDLE;
    VkCommandPool         cmdPool_   = VK_NULL_HANDLE;
    VkCommandBuffer       cmdBuf_    = VK_NULL_HANDLE;
    VkFence               fence_     = VK_NULL_HANDLE;
};

// ============================================================
// Main
// ============================================================

int main() {
    const uint32_t n = N;
    std::vector<float> A(n * n), B(n * n), C(n * n);
    fill_random(A);
    fill_random(B);

    int nthreads = NTHREADS > 0
                   ? NTHREADS
                   : (int)std::thread::hardware_concurrency();
    if (nthreads < 1) nthreads = 4;

    std::cout << "Matrix size: " << n << " x " << n << "\n";
    std::cout << "Threads (pthread): " << nthreads << "\n\n";

    // --- Function 1: Naive ---
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_naive(A.data(), B.data(), C.data(), n);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "Naive:  " << ms << " ms\n";
    }

    // --- Function 2: AVX2 ---
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_avx2(A.data(), B.data(), C.data(), n);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "AVX2:   " << ms << " ms\n";
    }

    // --- Function 4: pthread ---
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_pthread(A.data(), B.data(), C.data(), n);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "pthread: " << ms << " ms\n";
    }

    // --- Function 5: pthread + AVX2 ---
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_pthread_avx2(A.data(), B.data(), C.data(), n);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "pthread+AVX2: " << ms << " ms\n";
    }

    // --- Function 3: Vulkan ---
    // Vulkan init is outside the timed region so we measure only compute.
    try {
        VulkanMatMul vk(n, "matmul.spv");

        auto t0 = std::chrono::high_resolution_clock::now();
        vk.compute(A.data(), B.data(), C.data());
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "Vulkan: " << ms << " ms  (upload + dispatch + download)\n";
    } catch (const std::exception& e) {
        std::cerr << "Vulkan error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
