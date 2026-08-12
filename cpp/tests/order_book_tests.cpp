#include "mm/order_book.hpp"

#include <cassert>
#include <iostream>

using mm::LimitOrderBook;
using mm::Side;

void price_time_priority_is_preserved() {
  LimitOrderBook book;
  book.add_limit(1, Side::Sell, 100, 5);
  book.add_limit(2, Side::Sell, 100, 4);
  const auto trades = book.add_limit(3, Side::Buy, 100, 7);

  assert(trades.size() == 2);
  assert(trades[0].maker_order_id == 1 && trades[0].price == 100 && trades[0].quantity == 5);
  assert(trades[1].maker_order_id == 2 && trades[1].price == 100 && trades[1].quantity == 2);
  assert(book.quantity_at(Side::Sell, 100).value() == 2);
  assert(!book.contains(1));
  assert(book.contains(2));
}

void cancel_removes_only_the_requested_order() {
  LimitOrderBook book;
  book.add_limit(10, Side::Buy, 99, 3);
  book.add_limit(11, Side::Buy, 99, 4);
  assert(book.cancel(10));
  assert(!book.cancel(10));
  assert(book.quantity_at(Side::Buy, 99).value() == 4);
  assert(book.contains(11));
}

void market_orders_do_not_rest() {
  LimitOrderBook book;
  book.add_limit(20, Side::Sell, 101, 2);
  const auto trades = book.add_market(21, Side::Buy, 5);
  assert(trades.size() == 1);
  assert(trades[0].maker_order_id == 20 && trades[0].quantity == 2);
  assert(!book.best_ask().has_value());
  assert(!book.contains(21));
}

int main() {
  price_time_priority_is_preserved();
  cancel_removes_only_the_requested_order();
  market_orders_do_not_rest();
  std::cout << "order_book_tests passed\n";
}
