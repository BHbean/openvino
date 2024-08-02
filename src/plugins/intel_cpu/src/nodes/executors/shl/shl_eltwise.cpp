#include "shl_eltwise.hpp"
#include "shl_utils.hpp"
#include "csinn/csi_nn.h"
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {

inline void log_unsupported_prec(const std::vector<MemoryDescPtr>& srcDescs,
                                 const std::vector<MemoryDescPtr>& dstDescs,
                                 const Algorithm eltwiseAlgorithm) {
    std::string srcPrec;
    for (size_t i = 0; i < srcDescs.size(); i++) {
        srcPrec += srcDescs[i]->getPrecision().to_string() + " ";
    }
    DEBUG_LOG(algToString(eltwiseAlgorithm), ": provided combination of src precisions: [", srcPrec,
                          "] and dst precision: ", dstDescs[0]->getPrecision().to_string(), " is not supported");
}

bool ShlEltwiseExecutor::isEltwiseAlgorithmSupported(Algorithm algorithm) {
    if (one_of(algorithm, Algorithm::EltwiseAdd,
                          Algorithm::EltwiseSubtract,
                          Algorithm::EltwiseMultiply,
                          Algorithm::EltwiseDivide,
                          Algorithm::EltwiseMaximum,
                          Algorithm::EltwiseMinimum,
                          Algorithm::EltwiseExp,
                          Algorithm::EltwiseClamp,
                          Algorithm::EltwiseRelu,
                          Algorithm::EltwisePrelu)) {
        return true;
    }
    return false;
}

bool ShlEltwiseExecutorBuilder::isSupported(const EltwiseAttrs& eltwiseAttrs,
                                            const std::vector<MemoryDescPtr>& srcDescs,
                                            const std::vector<MemoryDescPtr>& dstDescs) const {
    if (!ShlEltwiseExecutor::isEltwiseAlgorithmSupported(eltwiseAttrs.algorithm)) {
        DEBUG_LOG("Eltwise algorithm ", algToString(eltwiseAttrs.algorithm), " is not supported");
        return false;
    }

    constexpr auto supported_prec = ov::element::f32;
    auto is_precision_supported = [supported_prec](const MemoryDescPtr& desc) { return desc->getPrecision() == supported_prec; };
    if (!(std::all_of(srcDescs.cbegin(), srcDescs.cend(), is_precision_supported) &&
          std::all_of(dstDescs.cbegin(), dstDescs.cend(), is_precision_supported))) {
        DEBUG_LOG("ShlEltwise supports only f32");
        return false;
    }

    for (const auto& srcDesc : srcDescs) {
        csinn_layout_enum supportedLayout = getShlDataLayoutByMemoryDesc(srcDesc);
        switch (eltwiseAttrs.algorithm) {
            case Algorithm::EltwisePrelu:
                // SHL PRelu op only supports these two kinds of layout
                if (!(supportedLayout == csinn_layout_enum::CSINN_LAYOUT_NC1HWC0 || supportedLayout == csinn_layout_enum::CSINN_LAYOUT_NCHW)) {
                    DEBUG_LOG("src descriptor layout is unsupported by SHL Prelu op: ", srcDesc->serializeFormat());
                    return false;
                }
                break;
            default:
                if (supportedLayout == csinn_layout_enum::CSINN_LAYOUT_NULL) {
                    DEBUG_LOG("src descriptor layout is unsupported by SHL: ", srcDesc->serializeFormat());
                    return false;
                }
                continue;
        }
    }
    for (const auto& dstDesc : dstDescs) {
        if (getShlDataLayoutByMemoryDesc(dstDesc) == csinn_layout_enum::CSINN_LAYOUT_NULL) {
            DEBUG_LOG("dst descriptor layout is unsupported by SHL: ", dstDesc->serializeFormat());
            return false;
        }
    }

    return true;
}

ShlEltwiseExecutor::ShlEltwiseExecutor(const ExecutorContext::CPtr context) : EltwiseExecutor(context) {}

bool ShlEltwiseExecutor::init(const EltwiseAttrs &eltwiseAttrs,
                              const std::vector<MemoryDescPtr> &srcDescs,
                              const std::vector<MemoryDescPtr> &dstDescs,
                              const std::vector<EltwisePostOp> &postOps) {
    if (!postOps.empty()) { return false; }
    shlEltwiseAttrs = eltwiseAttrs;

    srcTensors = std::vector<ShlTensor>(srcDescs.size());
    dstTensors = std::vector<ShlTensor>(dstDescs.size());

    // Allocate Shl session
    sess = ShlSession();

    for (size_t i = 0; i < srcDescs.size(); i++) {
        srcTensors[i] = ShlTensor(sess, precisionToShlDataType(srcDescs[i]->getPrecision()), getShlDataLayoutByMemoryDesc(srcDescs[i]), srcDescs[i]->getShape().getStaticDims());
    }
    for (size_t i = 0; i < dstDescs.size(); i++) {
        dstTensors[i] = ShlTensor(sess, precisionToShlDataType(dstDescs[i]->getPrecision()), getShlDataLayoutByMemoryDesc(dstDescs[i]), dstDescs[i]->getShape().getStaticDims());
    }

    std::function<int()> initFunc = nullptr;
    enum csinn_api_enum shl_api = CSINN_RVV;
    switch (shlEltwiseAttrs.algorithm) {
    case Algorithm::EltwiseAdd:
        params = ov::intel_cpu::make_unique<ShlDisoParams>(sess, shl_api);
        initFunc = [&]() {
            return csinn_add_init(srcTensors[0].get(), srcTensors[1].get(), dstTensors[0].get(), static_cast<csinn_diso_params*>(params->get()));
        };
        shlExecFunc = [&]() {
            return csinn_add(srcTensors[0].get(), srcTensors[1].get(), dstTensors[0].get(), static_cast<csinn_diso_params*>(params->get()));
        };
        break;
    case Algorithm::EltwiseSubtract:
        params = ov::intel_cpu::make_unique<ShlDisoParams>(sess, shl_api);
        initFunc = [&]() {
            return csinn_sub_init(srcTensors[0].get(), srcTensors[1].get(), dstTensors[0].get(), static_cast<csinn_diso_params*>(params->get()));
        };
        shlExecFunc = [&]() {
            return csinn_sub(srcTensors[0].get(), srcTensors[1].get(), dstTensors[0].get(), static_cast<csinn_diso_params*>(params->get()));
        };
        break;
    case Algorithm::EltwiseMultiply:
        params = ov::intel_cpu::make_unique<ShlDisoParams>(sess, shl_api);
        initFunc = [&]() {
            return csinn_mul_init(srcTensors[0].get(), srcTensors[1].get(), dstTensors[0].get(), static_cast<csinn_diso_params*>(params->get()));
        };
        shlExecFunc = [&]() {
            return csinn_mul(srcTensors[0].get(), srcTensors[1].get(), dstTensors[0].get(), static_cast<csinn_diso_params*>(params->get()));
        };
        break;
    case Algorithm::EltwiseDivide:
        params = ov::intel_cpu::make_unique<ShlDisoParams>(sess, shl_api);
        initFunc = [&]() {
            return csinn_div_init(srcTensors[0].get(), srcTensors[1].get(), dstTensors[0].get(), static_cast<csinn_diso_params*>(params->get()));
        };
        shlExecFunc = [&]() {
            return csinn_div(srcTensors[0].get(), srcTensors[1].get(), dstTensors[0].get(), static_cast<csinn_diso_params*>(params->get()));
        };
        break;
    case Algorithm::EltwiseMaximum:
        params = ov::intel_cpu::make_unique<ShlDisoParams>(sess, shl_api);
        initFunc = [&]() {
            return csinn_maximum_init(srcTensors[0].get(), srcTensors[1].get(), dstTensors[0].get(), static_cast<csinn_diso_params*>(params->get()));
        };
        shlExecFunc = [&]() {
            return csinn_maximum(srcTensors[0].get(), srcTensors[1].get(), dstTensors[0].get(), static_cast<csinn_diso_params*>(params->get()));
        };
        break;
    case Algorithm::EltwiseMinimum:
        params = ov::intel_cpu::make_unique<ShlDisoParams>(sess, shl_api);
        initFunc = [&]() {
            return csinn_minimum_init(srcTensors[0].get(), srcTensors[1].get(), dstTensors[0].get(), static_cast<csinn_diso_params*>(params->get()));
        };
        shlExecFunc = [&]() {
            return csinn_minimum(srcTensors[0].get(), srcTensors[1].get(), dstTensors[0].get(), static_cast<csinn_diso_params*>(params->get()));
        };
        break;
        // return true;
    case Algorithm::EltwiseExp:
        params = ov::intel_cpu::make_unique<ShlSisoParams>(sess, shl_api);
        initFunc = [&]() {
            return csinn_exp_init(srcTensors[0].get(), dstTensors[0].get(), static_cast<csinn_siso_params*>(params->get()));
        };
        shlExecFunc = [&]() {
            return csinn_exp(srcTensors[0].get(), dstTensors[0].get(), static_cast<csinn_siso_params*>(params->get()));
        };
        break;
    case Algorithm::EltwiseClamp:
        params = ov::intel_cpu::make_unique<ShlClipParams>(sess, shl_api, eltwiseAttrs.alpha, eltwiseAttrs.beta);
        initFunc = [&]() {
            return csinn_clip_init(srcTensors[0].get(), dstTensors[0].get(), static_cast<csinn_clip_params*>(params->get()));
        };
        shlExecFunc = [&]() {
            return csinn_clip(srcTensors[0].get(), dstTensors[0].get(), static_cast<csinn_clip_params*>(params->get()));
        };
        break;
    case Algorithm::EltwiseRelu:
        if (shlEltwiseAttrs.alpha == 0) {
            params = ov::intel_cpu::make_unique<ShlReluParams>(sess, shl_api);
            initFunc = [&]() {
                return csinn_relu_init(srcTensors[0].get(), dstTensors[0].get(), static_cast<csinn_relu_params*>(params->get()));
            };
            shlExecFunc = [&]() {
                return csinn_relu(srcTensors[0].get(), dstTensors[0].get(), static_cast<csinn_relu_params*>(params->get()));
            };
        } else {
            params = ov::intel_cpu::make_unique<ShlReluParams>(sess, shl_api, eltwiseAttrs.alpha);
            initFunc = [&]() {
                return csinn_leaky_relu_init(srcTensors[0].get(), dstTensors[0].get(), static_cast<csinn_relu_params*>(params->get()));
            };
            shlExecFunc = [&]() {
                return csinn_leaky_relu(srcTensors[0].get(), dstTensors[0].get(), static_cast<csinn_relu_params*>(params->get()));
            };
        }
        break;
    case Algorithm::EltwisePrelu:
        params = ov::intel_cpu::make_unique<ShlPReluParams>(sess, shl_api);
        initFunc = [&]() {
            return csinn_prelu_init(srcTensors[0].get(), srcTensors[1].get(), dstTensors[0].get(), static_cast<csinn_prelu_params*>(params->get()));
        };
        shlExecFunc = [&]() {
            return csinn_prelu(srcTensors[0].get(), srcTensors[1].get(), dstTensors[0].get(), static_cast<csinn_prelu_params*>(params->get()));
        };
        break;
    default:
        OPENVINO_THROW("Unsupported operation type for SHL Eltwise executor: ",
                       static_cast<int>(shlEltwiseAttrs.algorithm));
    }

    return initFunc != nullptr && initFunc() == CSINN_TRUE;
}

void ShlEltwiseExecutor::exec(const std::vector<MemoryCPtr> &src,
                              const std::vector<MemoryPtr> &dst,
                              const void *post_ops_data_) {
    for (size_t i = 0; i < src.size(); i++) {
        srcTensors[i].setData(src[i]->getData());
    }
    for (size_t i = 0; i < dst.size(); i++) {
        dstTensors[i].setData(dst[i]->getData());
    }
    // std::cout << shlExecFunc() << std::endl;
    OPENVINO_ASSERT(shlExecFunc != nullptr && shlExecFunc() == CSINN_TRUE,
                    "ShlEltwiseExecutor: failed to execute");

    return;
}

}   // namespace intel_cpu
}   // namespace ov