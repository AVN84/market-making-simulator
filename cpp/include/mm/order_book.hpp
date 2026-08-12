#pragma once

#include <cstdint>
#include <list>
#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

namespace mm {

enum class Side { Buy, Sell };

using OrderId = std::uint64_t;
using Price = std::int64_t;
using Quantity = std::int64_t;

struct Order {
  OrderId id;
  Side side;
  Price price;
  Quantity quantity;
  std::uint64_t sequence;
};

struct Trade {
  OrderId maker_order_id;
  OrderId taker_order_id;
  Price price;
  Quantity quantity;
};

// A price-time-priority central limit order book. Prices are integer ticks.
class LimitOrderBook {
 public:
  std::vector<Trade> add_limit(OrderId id, Side side, Price price, Quantity quantity);
  std::vector<Trade> add_market(OrderId id, Side side, Quantity quantity);
  bool cancel(OrderId id);

  [[nodiscard]] std::optional<Price> best_bid() const;
  [[nodiscard]] std::optional<Price> best_ask() const;
  [[nodiscard]] std::optional<Quantity> quantity_at(Side side, Price price) const;
  [[nodiscard]] bool contains(OrderId id) const;

 private:
  using Level = std::list<Order>;
  using BidLevels = std::map<Price, Level, std::greater<Price>>;
  using AskLevels = std::map<Price, Level, std::less<Price>>;

  struct Locator {
    Side side;
    Price price;
    Level::iterator order;
  };

  std::vector<Trade> match(OrderId taker_id, Side side, Price limit, Quantity& remaining);
  void rest(OrderId id, Side side, Price price, Quantity quantity);
  void erase_front(Side resting_side, Price price);

  BidLevels bids_;
  AskLevels asks_;
  std::unordered_map<OrderId, Locator> locations_;
  std::uint64_t next_sequence_ = 1;
};

}  // namespace mm
