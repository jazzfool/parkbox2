#pragma once

#include "delegate.hpp"

#include <functional>
#include <optional>
#include <unordered_map>

template <typename... Ts>
class Signal;

template <typename... Ts>
struct SignalListener {
  private:
    friend Signal<Ts...>;

    SignalListener(std::size_t idx) : idx{idx} {
    }

    std::size_t idx;
};

template <typename... Ts>
class Signal {
  public:
    using DelegateT = Delegate<void(Ts...)>;
    using FunctorT = std::function<void(Ts...)>;
    using SignalListenerT = SignalListener<Ts...>;

    SignalListenerT connect_delegate(DelegateT delegate) {
        const std::size_t id = nextID++;
        delegates.emplace(id, delegate);
        return {id};
    }

    template <auto F>
    SignalListenerT connect_delegate(typename detail::DelegateInspector<F>::ObjectT* o) {
        return connect_delegate(delegate<F>(o));
    }

    template <auto F>
    SignalListenerT connect_delegate(typename detail::DelegateInspector<F>::ObjectT& o) {
        return connect_delegate(delegate<F>(o));
    }

    void disconnect(SignalListenerT listener) {
        delegates.erase(listener.idx);
    }

    template <typename... Args>
    void operator()(Args&&... arg) const {
        for (const auto& delegate : delegates) {
            if (delegate.second)
                (delegate.second)(std::forward<Args>(arg)...);
        }
    }

  private:
    std::size_t nextID = 0;
    std::unordered_map<std::size_t, DelegateT> delegates;
};

template <typename... Ts>
class ScopedSignalListener {
  public:
    ScopedSignalListener() : signal{nullptr} {
    }

    ScopedSignalListener(Signal<Ts...>& signal, SignalListener<Ts...> listener) : signal{&signal}, listener{listener} {
    }

    ScopedSignalListener(ScopedSignalListener&& other) : signal{std::exchange(other.signal, nullptr)}, listener{std::move(other.signal)} {
    }

    ScopedSignalListener(const ScopedSignalListener&) = delete;

    ~ScopedSignalListener() {
        if (*this) {
            signal->disconnect(*listener);
        }
    }

    ScopedSignalListener& operator=(ScopedSignalListener&& rhs) {
        if (this != &rhs) {
            signal = std::exchange(rhs.signal, nullptr);
            listener = std::move(rhs.listener);
        }

        return *this;
    }

    ScopedSignalListener& operator=(const ScopedSignalListener&) = delete;

    operator bool() const {
        return signal != nullptr && listener.has_value();
    }

  private:
    Signal<Ts...>* signal;
    std::optional<SignalListener<Ts...>> listener;
};
