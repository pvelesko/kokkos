#ifndef KOKKOS_TYPECHECKS_HPP
#define KOKKOS_TYPECHECKS_HPP

#include <type_traits>
template <class Obj>
void isTriviallyCopyable() {
  static_assert(std::is_trivially_copyable<Obj>::value);

  static_assert(std::is_copy_constructible<Obj>::value ||
                    std::is_move_constructible<Obj>::value ||
                    std::is_copy_assignable<Obj>::value ||
                    std::is_move_assignable<Obj>::value,
                "Obj copy/move constructors/assignments deleted");

  static_assert(std::is_trivially_copy_constructible<Obj>::value ||
                    !std::is_copy_constructible<Obj>::value,
                "Obj not trivially copy constructible");

  static_assert(std::is_trivially_move_constructible<Obj>::value ||
                    !std::is_move_constructible<Obj>::value,
                "Obj not trivially move constructible");

  static_assert(std::is_trivially_copy_assignable<Obj>::value ||
                    !std::is_copy_assignable<Obj>::value,
                "Obj not trivially copy assignable");

  static_assert(std::is_trivially_move_assignable<Obj>::value ||
                    !std::is_move_assignable<Obj>::value,
                "Obj not trivially move assignable");

  static_assert(std::is_trivially_destructible<Obj>::value,
                "Obj not trivially destructible");
}
#endif
