#pragma once

#include "def.hpp"

#include <utility>

template <typename T>
class Delegate {};

template <typename T, typename... Ts>
class Delegate<T(Ts...)> {
  public:
    using FuncT = T(void*, Ts...);

    Delegate() : obj{nullptr}, func{nullptr} {
    }

    Delegate(void* obj, FuncT* func) : obj{obj}, func{func} {
    }

    template <typename... Args, typename = std::enable_if_t<sizeof...(Ts) == sizeof...(Args)>>
    T operator()(Args&&... arg) const {
        PK_ASSERT((*this || std::is_same_v<T, void>));
        if (*this)
            return func(obj, std::forward<Args>(arg)...);
    }

    operator bool() const {
        return obj != nullptr && func != nullptr;
    }

  private:
    void* obj;
    FuncT* func;
};

namespace detail {

template <typename T, T V>
struct DelegateInspectorT {};

template <typename T, typename R, typename... Ts, R (T::*F)(Ts...)>
struct DelegateInspectorT<R (T::*)(Ts...), F> {
    using DelegateT = Delegate<R(Ts...)>;
    using ObjectT = T;

    static R invoke(void* obj, Ts... arg) {
        return (((T*)obj)->*F)(arg...);
    }
};

template <auto F>
using DelegateInspector = DelegateInspectorT<decltype(F), F>;

} // namespace detail

template <auto F>
typename detail::DelegateInspector<F>::DelegateT delegate(typename detail::DelegateInspector<F>::ObjectT* o) {
    return typename detail::DelegateInspector<F>::DelegateT{o, &detail::DelegateInspector<F>::invoke};
}

template <auto F>
typename detail::DelegateInspector<F>::DelegateT delegate(typename detail::DelegateInspector<F>::ObjectT& o) {
    return delegate<F>(&o);
}
