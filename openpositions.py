from ib_insync import IB, util, Stock
import pandas as pd

def main():
    ib = IB()
    ib.connect('127.0.0.1', 4001, clientId=1)
    print("Connected.")

    positions = ib.positions()
    trades = ib.trades()

    # Map for quick lookup of stop orders by symbol
    stop_orders = {}
    for trade in trades:
        order = trade.order
        contract = trade.contract
        if order.orderType == 'STP' and order.tif == 'GTC' and order.action == 'SELL':
            stop_orders[contract.symbol] = order.auxPrice  # Stop level

    print(f"{'Symbol':6} | {'Qty':>6} | {'AvgPx':>8} | {'StopLoss(GTC)':>13}")
    print("-" * 45)

    for pos in positions:
        contract = pos.contract
        symbol = contract.symbol
        qty = pos.position
        avg_cost = pos.avgCost
        stop = stop_orders.get(symbol, "None")
        print(f"{symbol:6} | {qty:6} | {avg_cost:8.2f} | {stop:>13}")

    ib.disconnect()

if __name__ == "__main__":
    main()
