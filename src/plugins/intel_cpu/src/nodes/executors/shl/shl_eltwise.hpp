#include "shl.hpp"
#include "cpu_memory.h"
#include "nodes/executors/eltwise.hpp"
#include <functional>
#include <tuple>
#include <utility>
#include <type_traits>

namespace ov {
namespace intel_cpu {

// implement `make_index_sequence` by hand
template<std::size_t... Indices>
struct index_sequence {};

template<std::size_t N, std::size_t... Indices>
struct make_index_sequence_impl : make_index_sequence_impl<N - 1, N - 1, Indices...> {};

template<std::size_t... Indices>
struct make_index_sequence_impl<0, Indices...> {
    using type = index_sequence<Indices...>;
};

template<std::size_t N>
using make_index_sequence = typename make_index_sequence_impl<N>::type;

class ShlEltwiseExecutor : public EltwiseExecutor {
public:
    explicit ShlEltwiseExecutor(const ExecutorContext::CPtr context);
    static bool isEltwiseAlgorithmSupported(Algorithm algorithm);

    bool init(const EltwiseAttrs& eltwiseAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const std::vector<EltwisePostOp>& postOps) override;

    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void *post_ops_data_) override;

    impl_desc_type getImplType() const override {
        return implType;
    }

    template<typename Func, typename... Args>
    void setFunc(Func&& initFunc, Func&& execFunc, Args&&... args) {
        init_func = [this, initFunc, args...]() { callFunc(initFunc, std::make_tuple(args...)); };
        exec_func = [this, execFunc, args...]() { callFunc(execFunc, std::make_tuple(args...)); };
    }


private:
    EltwiseAttrs shlEltwiseAttrs{};
    impl_desc_type implType = impl_desc_type::shl;
    ShlSession sess = {};
    std::vector<ShlTensor> srcTensors, dstTensors;
    std::unique_ptr<IShlParams> params;
    std::function<void()> init_func;
    std::function<void()> exec_func;

    template<typename Func, typename Tuple, size_t... Index>
    void callFunc(Func&& func, Tuple&& tuple, index_sequence<Index...>) const {
        func(std::get<Index>(std::forward<Tuple>(tuple))...);
    }

    template<typename Func, typename Tuple>
    void callFunc(Func&& func, Tuple&& tuple) const {
        callFunc(std::forward<Func>(func), std::forward<Tuple>(tuple),
                 make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>{});
    }
};

class ShlEltwiseExecutorBuilder : public EltwiseExecutorBuilder {
public:
    bool isSupported(const EltwiseAttrs& eltwiseAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override;

    EltwiseExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<ShlEltwiseExecutor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov