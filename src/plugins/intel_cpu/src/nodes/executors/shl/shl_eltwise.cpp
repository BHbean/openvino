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
    if (one_of(algorithm, Algorithm::EltwiseAdd)) {
        return true;
    }
    return false;
}

bool ShlEltwiseExecutorBuilder::isSupported(const EltwiseAttrs& eltwiseAttrs,
                                            const std::vector<MemoryDescPtr>& srcDescs,
                                            const std::vector<MemoryDescPtr>& dstDescs) const {
    auto checkPrecision = [&srcDescs, &dstDescs](std::vector<ov::element::Type> srcVecPrc, ov::element::Type dstPrc) -> bool {
        for (size_t i = 0; i < srcDescs.size(); i++) {
            if (srcDescs[i]->getPrecision() != srcVecPrc[i]) return false;
        }
        if (dstDescs[0]->getPrecision() != dstPrc) { return false; }
        return true;
    };

    switch (eltwiseAttrs.algorithm) {
        // Support eltwise ops with `FP32` precision only for now 
        case Algorithm::EltwiseAdd:
        case Algorithm::EltwiseSubtract:
        case Algorithm::EltwiseMultiply:
        case Algorithm::EltwiseDivide:
        case Algorithm::EltwiseExp:
        case Algorithm::EltwiseRelu:
            if (!(checkPrecision({ov::element::f32, ov::element::f32}, ov::element::f32))) {
                log_unsupported_prec(srcDescs, dstDescs, eltwiseAttrs.algorithm);
                return false;
            }
            break;
        default:
            DEBUG_LOG("Eltwise algorithm ", algToString(eltwiseAttrs.algorithm), " is not supported");
            return false;
    }

    for (const auto & srcDesc : srcDescs) {
        if (getShlDataLayoutByMemoryDesc(srcDesc) == csinn_layout_enum::CSINN_LAYOUT_NULL) {
            DEBUG_LOG("src descriptor layout is unsupported by SHL: ", srcDesc->serializeFormat());
            return false;
        }
    }
    for (const auto & dstDesc : dstDescs) {
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
    sess = ShlSession(CSINN_RM_LAYER);

    for (size_t i = 0; i < srcDescs.size(); i++) {
        srcTensors[i] = ShlTensor(sess, precisionToShlDataType(srcDescs[i]->getPrecision()), getShlDataLayoutByMemoryDesc(srcDescs[i]));
    }
    for (size_t i = 0; i < dstDescs.size(); i++) {
        dstTensors[i] = ShlTensor(sess, precisionToShlDataType(dstDescs[i]->getPrecision()), getShlDataLayoutByMemoryDesc(dstDescs[i]));
    }

    switch (shlEltwiseAttrs.algorithm) {
    case Algorithm::EltwiseAdd:
        params = ov::intel_cpu::make_unique<ShlDisoParams>(sess, CSINN_RVV);
        setFunc(csinn_add_init, csinn_add, srcTensors[0].get(), srcTensors[1].get(), dstTensors[0].get(), static_cast<csinn_diso_params*>(params->get()));
        break;
    case Algorithm::EltwiseSubtract:
        params = ov::intel_cpu::make_unique<ShlDisoParams>(sess, CSINN_RVV);
        setFunc(csinn_sub_init, csinn_sub, srcTensors[0].get(), srcTensors[1].get(), dstTensors[0].get(), static_cast<csinn_diso_params*>(params->get()));
        break;
    case Algorithm::EltwiseMultiply:
        params = ov::intel_cpu::make_unique<ShlDisoParams>(sess, CSINN_RVV);
        setFunc(csinn_mul_init, csinn_mul, srcTensors[0].get(), srcTensors[1].get(), dstTensors[0].get(), static_cast<csinn_diso_params*>(params->get()));
        break;
    case Algorithm::EltwiseDivide:
        params = ov::intel_cpu::make_unique<ShlDisoParams>(sess, CSINN_RVV);
        setFunc(csinn_div_init, csinn_div, srcTensors[0].get(), srcTensors[1].get(), dstTensors[0].get(), static_cast<csinn_diso_params*>(params->get()));
        break;
    case Algorithm::EltwiseExp:
        params = ov::intel_cpu::make_unique<ShlSisoParams>(sess, CSINN_RVV);
        setFunc(csinn_exp_init, csinn_exp, srcTensors[0].get(), dstTensors[0].get(), static_cast<csinn_siso_params*>(params->get()));
        break;
    case Algorithm::EltwiseRelu:
        params = ov::intel_cpu::make_unique<ShlReluParams>(sess, CSINN_RVV);
        setFunc(csinn_relu_init, csinn_relu, srcTensors[0].get(), dstTensors[0].get(), static_cast<csinn_relu_params*>(params->get()));
        break;
    default:
        OPENVINO_THROW("Unsupported operation type for SHL Eltwise executor: ",
                       static_cast<int>(shlEltwiseAttrs.algorithm));
    }

    return init_func != nullptr && init_func() == CSINN_TRUE;
}

void ShlEltwiseExecutor::exec(const std::vector<MemoryCPtr> &src,
                              const std::vector<MemoryPtr> &dst,
                              const void *post_ops_data_) {
    for (size_t i = 0; i < src.size(); i++) {
        srcTensors[i].setData(src[i]->getData());
        srcTensors[i].setShape(src[i]->getDescPtr()->getShape().getStaticDims());
    }
    for (size_t i = 0; i < dst.size(); i++) {
        dstTensors[i].setData(dst[i]->getData());
        dstTensors[i].setShape(dst[i]->getDescPtr()->getShape().getStaticDims());
    }

    OPENVINO_ASSERT(exec_func != nullptr && exec_func() == CSINN_TRUE,
                    "ShlEltwiseExecutor: failed to execute");

    return;
}

}   // namespace intel_cpu
}   // namespace ov