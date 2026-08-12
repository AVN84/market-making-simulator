#include "mm/order_book.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace mm {

namespace {

void validate_new_order(OrderId id, Price price, Quantity quantity, bool exists) {
  if (id == 0) {
    throw std::invalid_argument("order id must be non-zero");
  }
  if (quantity <= 0) {
    throw std::invalid_argument("order quantity must be positive");
  }
  if (price < 0) {
    throw std::invalid_argument("order price must be non-negative");
  }
  if (exists) {
    throw std::invalid_argument("duplicate live order id");
  }
}

}  // namespace

std::vector<Trade> LimitOrderBook::add_limit(OrderId id, Side side, Price price,
                                              Quantity quantity) {
  validate_new_order(id, price, quantity, contains(id));
  Quantity remaining = quantity;
  auto trades = match(id, side, price, remaining);
  if (remaining > 0) {
    rest(id, side, price, remaining);
  }
  return trades;
}

std::vector<Trade> LimitOrderBook::add_market(OrderId id, Side side, Quantity quantity) {
  validate_new_order(id, 0, quantity, contains(id));
  Quantity remaining = quantity;
  const Price unbounded = side == Side::Buy ? std::numeric_limits<Price>::max() : 0;
  return match(id, side, unbounded, remaining);
}

std::vector<Trade> LimitOrderBook::match(OrderId taker_id, Side side, Price limit,
                                          Quantity& remaining) {
  std::vector<Trade> trades;
  if (side == Side::Buy) {
    while (remaining > 0 && !asks_.empty() && asks_.begin()->first <= limit) {
      const Price price = asks_.begin()->first;
      Order& maker = asks_.begin()->second.front();
      const Quantity executed = std::min(remaining, maker.quantity);
      trades.push_back({maker.id, taker_id, price, executed});
      maker.quantity -= executed;
      remaining -= executed;
      if (maker.quantity == 0) {
        erase_front(Side::Sell, price);
      }
    }
  } else {
    while (remaining > 0 && !bids_.empty() && bids_.begin()->first >= limit) {
      const Price price = bids_.begin()->first;
      Order& maker = bids_.begin()->second.front();
      const Quantity executed = std::min(remaining, maker.quantity);
      trades.push_back({maker.id, taker_id, price, executed});
      maker.quantity -= executed;
      remaining -= executed;
      if (maker.quantity == 0) {
        erase_front(Side::Buy, price);
      }
    }
  }
  return trades;
}

void LimitOrderBook::rest(OrderId id, Side side, Price price, Quantity quantity) {
  Order order{id, side, price, quantity, next_sequence_++};
  if (side == Side::Buy) {
    auto& level = bids_[price];
    level.push_back(order);
    locations_.emplace(id, Locator{side, price, std::prev(level.end())});
  } else {
    auto& level = asks_[price];
    level.push_back(order);
    locations_.emplace(id, Locator{side, price, std::prev(level.end())});
  }
}

void LimitOrderBook::erase_front(Side resting_side, Price price) {
  if (resting_side == Side::Buy) {
    auto level = bids_.find(price);
    locations_.erase(level->second.front().id);
    level->second.pop_front();
    if (level->second.empty()) {
      bids_.erase(level);
    }
  } else {
    auto level = asks_.find(price);
    locations_.erase(level->second.front().id);
    level->second.pop_front();
    if (level->second.empty()) {
      asks_.erase(level);
    }
  }
}

bool LimitOrderBook::cancel(OrderId id) {
  auto location = locations_.find(id);
  if (location == locations_.end()) {
    return false;
  }
  const Locator locator = location->second;
  if (locator.side == Side::Buy) {
    auto level = bids_.find(locator.price);
    level->second.erase(locator.order);
    if (level->second.empty()) {
      bids_.erase(level);
    }
  } else {
    auto level = asks_.find(locator.price);
    level->second.erase(locator.order);
    if (level->second.empty()) {
      asks_.erase(level);
    }
  }
  locations_.erase(location);
  return true;
}

std::optional<Price> LimitOrderBook::best_bid() const {
  if (bids_.empty()) return std::nullopt;
  return bids_.begin()->first;
}

std::optional<Price> LimitOrderBook::best_ask() const {
  if (asks_.empty()) return std::nullopt;
  return asks_.begin()->first;
}

std::optional<Quantity> LimitOrderBook::quantity_at(Side side, Price price) const {
  Quantity total = 0;
  if (side == Side::Buy) {
    const auto level = bids_.find(price);
    if (level == bids_.end()) return std::nullopt;
    for (const auto& order : level->second) total += order.quantity;
  } else {
    const auto level = asks_.find(price);
    if (level == asks_.end()) return std::nullopt;
    for (const auto& order : level->second) total += order.quantity;
  }
  return total;
}

bool LimitOrderBook::contains(OrderId id) const { return locations_.contains(id); }

}  // namespace mm
