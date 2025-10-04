from ib_insync import *
ib = IB(); ib.connect('127.0.0.1', 4001, 101)
stock = Stock('AAPL', 'SMART', 'USD')
ib.qualifyContracts(stock)
chains = ib.reqSecDefOptParams('AAPL', '', 'STK', stock.conId)
print(chains)
