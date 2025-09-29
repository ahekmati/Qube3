from ib_insync import *

def main():
    # Connect to Interactive Brokers TWS or Gateway on port 4001
    ib = IB()
    ib.connect('127.0.0.1', 4001, clientId=1)

    # Get all open positions and open orders
    positions = ib.positions()
    open_orders = ib.reqAllOpenOrders()

    # Filter stop loss orders ('STP' or 'STP LMT')
    stop_orders = [o for o in open_orders if o.order.orderType.startswith('STP')]

    print("Open Positions and associated Stop Loss Orders:\n")
    for pos in positions:
        contract = pos.contract
        symbol = contract.localSymbol or f"{contract.symbol} {contract.secType}"
        pos_side = "Long" if pos.position > 0 else "Short"
        qty = abs(pos.position)
        avg_price = pos.avgCost if hasattr(pos, "avgCost") else "N/A"

        # Request current market price snapshot
        ticker_tick = ib.reqMktData(contract, '', False, False)
        ib.sleep(1)  # Give IB time to populate
        current_price = ticker_tick.last if ticker_tick.last else ticker_tick.close or "N/A"
        ib.cancelMktData(contract)

        matching_stop = [
            o for o in stop_orders
            if o.contract.conId == contract.conId
        ]

        print(f"{symbol}: {pos_side}, Qty: {qty}")
        print(f"  Bought at: {avg_price}")
        print(f"  Current price: {current_price}")
        if matching_stop:
            for stop in matching_stop:
                print(f"  -> STOP LOSS (orderId {stop.order.permId}, AuxPrice {stop.order.auxPrice})")
        else:
            print("  -> No stop loss order found")

    if not positions:
        print("No open positions found.")

    ib.disconnect()

if __name__ == "__main__":
    main()
