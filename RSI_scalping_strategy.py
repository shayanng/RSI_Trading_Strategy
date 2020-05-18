######## strategy info:
"""
this is a stand alone strategy working with RSI to generate signal on M5 ohlc data
"""


import oandapyV20
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import pandas as pd
import numpy as np
import schedule
import time
import matplotlib.pyplot as plt
plt.style.use("ggplot")


########################### API Connection ##############################

# Connecting to API
API_path = "C:\\Users\\naghi\\API keys\\Oanda_API.txt"
client = oandapyV20.API(access_token=open(API_path, "r").read(),
                        environment="practice")

account_id = "your account number"

########################### Data Generation ##############################

#tools for oanda data manupulations and order
def candles(instrument, n_candles: int = 250, timeframe: str = "H1"):
    """
    function to get data from Oanda API
    """
    params = {"count": n_candles, "granularity": timeframe}
    candles = instruments.InstrumentsCandles(instrument=instrument, params=params)
    client.request(candles)
    ohlc_dict = candles.response["candles"]
    ohlc = pd.DataFrame(ohlc_dict)
    ohlc_df = ohlc.mid.dropna().apply(pd.Series)
    ohlc_df["volume"] = ohlc["volume"]
    ohlc_df.index = ohlc["time"]
    ohlc_df = ohlc_df.apply(pd.to_numeric)
    return ohlc_df

####################### Order Commands Execution #######################

def market_order(instrument, units, sl, tp):
    """
    units can be positive or negative, stop loss (in pips) added/subtracted to price
    """
    account_ID = account_id
    data = {
        "order": {
            "price": "",
            "stopLossOnFill": {
                "trailingStopLossOnFill": "GTC",
                "distance": str(sl)},
            "takeProfitOnFill": {
                # "takeProfitOnFill": "GTC",
                "price": str(tp)},
            # "distance": str(tp)},
            "timeInForce": "FOK",
            "instrument": str(instrument),
            "units": str(units),
            "type": "MARKET",
            "positionFill": "DEFAULT"
        }
    }
    r = orders.OrderCreate(accountID=account_ID, data=data)
    client.request(r)

def market_order_sl_distance(instrument, units, sl, tp):
    """
    units can be positive or negative, stop loss (in pips) added/subtracted to price
    """
    account_ID = account_id
    data = {
        "order": {
            "price": "",
            "stopLossOnFill": {
                "trailingStopLossOnFill": "GTC",
                "distance": str(sl)},
            "takeProfitOnFill": {
                # "takeProfitOnFill": "GTC",
                "price": str(tp)},
            # "distance": str(tp)},
            "timeInForce": "FOK",
            "instrument": str(instrument),
            "units": str(units),
            "type": "MARKET",
            "positionFill": "DEFAULT"
        }
    }
    r = orders.OrderCreate(accountID=account_ID, data=data)
    client.request(r)


def market_order_sl_price(instrument, units, sl, tp):
    """
    units can be positive or negative, stop loss (in pips) added/subtracted to price
    """
    account_ID = account_id
    data = {
        "order": {
            "price": "",
            "stopLossOnFill": {
                # "trailingStopLossOnFill": "GTC",
                "timeInForce": "GTC",
                # "distance": str(sl)},
                "price": str(sl)},
            "takeProfitOnFill": {
                # "takeProfitOnFill": "GTC",
                "price": str(tp)},
            # "distance": str(tp)},
            "timeInForce": "FOK",
            "instrument": str(instrument),
            "units": str(units),
            "type": "MARKET",
            "positionFill": "DEFAULT"
        }
    }
    r = orders.OrderCreate(accountID=account_ID, data=data)
    client.request(r)


def limit_order(instrument, units, price, sl, tp):
    """
    units can be positive or negative, stop loss (in pips) added/subtracted to price
    """
    account_ID = account_id
    data = {
        "order": {
            "price": str(price),
            "stopLossOnFill": {
                # "trailingStopLossOnFill": "GTC",
                "timeInForce": "GTC",
                # "distance": str(sl)},
                "price": str(sl)},
            "takeProfitOnFill": {
                # "takeProfitOnFill": "GTC",
                "price": str(tp)},
            # "distance": str(tp)},
            "timeInForce": "GTC",
            "instrument": str(instrument),
            "units": str(units),
            "type": "LIMIT",
            "positionFill": "DEFAULT"
        }
    }
    r = orders.OrderCreate(accountID=account_ID, data=data)
    client.request(r)


#################### Customized ATR Generator (CAG) ####################

def get_ATR(DF, n):
    """
    function to calculate True Range and Average True Range
    """
    df = DF.copy()
    df['H-L'] = abs(df['h'] - df['l'])
    df['H-PC'] = abs(df['h'] - df['c'].shift(1))
    df['L-PC'] = abs(df['l'] - df['c'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    # df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    # df2 = df.drop(['H-L','H-PC','L-PC','TR'],axis=1)
    # return round(df2["ATR"][-1],2)
    return df["ATR"]


def get_atr_values(df, per1, per2, per3, per4):
    """
    calculates multiple ATR values for the data-frame.
    """
    df[f"ATR_{per1}"] = get_ATR(df, per1)  # ["ATR"]
    df[f"ATR_{per2}"] = get_ATR(df, per2)  # ["ATR"]
    df[f"ATR_{per3}"] = get_ATR(df, per3)  # ["ATR"]
    df[f"ATR_{per4}"] = get_ATR(df, per4)  # ["ATR"]
    df["iATR"] = ((df[f"ATR_{per1}"] * 1) + (df[f"ATR_{per2}"] * 2) + (df[f"ATR_{per3}"] * 3) + (
                df[f"ATR_{per4}"] * 4)) / 10
    return df


# data_daily = oanda.candles("EUR_USD", 800, "D")
# data_h1 = oanda.candles("EUR_USD", 800, "H1")
# data_daily_ATR = get_atr_values(data_daily, 5, 7, 55, 264)
# position_size = risk.position_sizing_rfp(data_daily_ATR, risk=0.01, sl_multiple=1)
# print(data_daily.index)

#################### Asset picking & Preparation #####################

# pairs for EU & US sessions
# pairs = ['EUR_USD', 'GBP_USD', 'USD_CHF', 'USD_CAD', "AUD_USD", "NZD_USD", "EUR_CAD"]
pairs = ['EUR_USD', 'GBP_USD', 'USD_CHF']

L1_signal_long = {}
L1_signal_short = {}
for i in pairs:
    L1_signal_long[i] = False
    L1_signal_short[i] = False

############################## RSI Engine #############################

# DF = oanda.candles("EUR_USD", 500, "H1")
# n = 8

def generate_RSI(DF, n):
    """
    calculates the Relative Strength Index (RSI)
    """
    df = DF.copy()
    df["delta"] = df["c"] - df["c"].shift(1)
    df["gain"] = np.where(df["delta"] >= 0, df["delta"], 0)
    df["loss"] = np.where(df["delta"] < 0, abs(df["delta"]), 0)
    avg_gain = []
    avg_loss = []
    gain = df["gain"].tolist()
    loss = df["loss"].tolist()
    for i in range(len(df)):
        if i < n:
            avg_gain.append(np.NaN)
            avg_loss.append(np.NaN)
        elif i == n:
            avg_gain.append(df["gain"].rolling(n).mean().tolist()[n])
            avg_loss.append(df["loss"].rolling(n).mean().tolist()[n])
        elif i > n:
            avg_gain.append(((n - 1)*avg_gain[i - 1] + gain[i]) / n)
            avg_loss.append(((n - 1)*avg_loss[i - 1] + loss[i]) / n)
    df["avg_gain"] = np.array(avg_gain)
    df["avg_loss"] = np.array(avg_loss)
    df["RS"] = df["avg_gain"]/df["avg_loss"]
    df["RSI"] = 100 - (100 / (1 + df["RS"]))
    df2 = df.drop(['delta','gain','loss', "avg_loss", "avg_gain", "RS"], axis=1)
    ### function visuals
    # plt.subplot(211)
    # plt.plot(df2.loc[:,['c']])
    # plt.title('close price chart')
    #
    # plt.subplot(212)
    # plt.plot(df.loc[:,["RSI"]], color="blue")
    # plt.hlines(y=20, xmin=0, xmax=len(df), linestyles='dashed')
    # plt.hlines(y=80, xmin=0, xmax=len(df), linestyles='dashed')
    # plt.show()
    return df2

###################### Signal Function ######################

def trade_signal(df, curr):
    """
    Returns the signal for the strategy
    """
    global L1_signal_long, L1_signal_short
    signal = ""

    # level 1 signal generation
    if df["RSI"][-1] > 80:
        L1_signal_long[curr] = False
        L1_signal_short[curr] = True
        print("RSI has reached and passed OVER-BOUGHT level")

    elif df["RSI"][-1] < 20:
        L1_signal_long[curr] = True
        L1_signal_short[curr] = False
        print("RSI has reached and passed OVER-SOLD level")

    else:
        L1_signal_long[curr] = False
        L1_signal_short[curr] = False
        print("condition level one has not yet been met")

    # level 2 signal generation
    if L1_signal_long[curr] == True:
        if df["RSI"][-1] < 20 and df["RSI"][-1] > 17.5:
            signal = "Buy"
            print("level 2 signal is to LONG.")
        elif df["RSI"][-1] < 17.5:
            signal = "Strong_Buy"
            print("level 2 signal is STRONG LONG")

    elif L1_signal_short[curr] == True:
        if df["RSI"][-1] > 80 and df["RSI"][-1] < 82.5:
            signal = "Sell"
            print("level 2 signal is to SHORT.")
        elif df["RSI"][-1] > 82.5:
            signal = "Strong_Sell"
            print("level 2 signal is STRONG SHORT")

    return signal


######## implementation of strategy with API main() #########


def main():
    global pairs
    try:
        r = trades.OpenTrades(accountID=account_id)
        open_trades = client.request(r)["trades"]
        curr_ls = []
        for i in range(len(open_trades)):
            curr_ls.append(open_trades[i]["instrument"])
        pairs = [i for i in pairs if i not in curr_ls]
        for currency in pairs:
            print(time.ctime())
            print("--------------------------")
            print("open positions: ", curr_ls)
            print("available currencies list: ", pairs)
            print("--------------------------")
            print("processing: ", currency)
            data = candles(currency, 300, "M5")
            data_RSI = generate_RSI(data, 10)
            data_ATR = get_atr_values(data_RSI, 5, 7, 55, 265)
            position_size = 20000
            signal = trade_signal(data_ATR, currency)
            print("ATR: ", round(data_ATR["iATR"][-1], 5))
            print("signal: ", signal)
            print("position size: ", position_size)
            print("..........................")

            # buy signals command
            if signal == "Buy":
                market_order_sl_price(currency, int(round(position_size * 0.75, 0)), round(data_ATR["c"][-1] - 0.00040, 5),
                                         round(0.00070 + data_ATR["c"][-1], 5))
                print("New long position initiated for ", currency)
                print("sl = ", round(data_ATR["c"][-1] - 0.00040, 5))
                print("tp = ", round(0.00070 + data_ATR["c"][-1], 5))

            if signal == "Strong_Buy":
                market_order_sl_price(currency, int(round(position_size * 1.2, 0)), round(data_ATR["c"][-1] - 0.00040, 5),
                                         round(0.00070 + data_ATR["c"][-1], 5))
                print("New long position initiated for ", currency)
                print("sl = ", round(data_ATR["c"][-1] - 0.00040, 5))
                print("tp = ", round(0.00070 + data_ATR["c"][-1], 5))

            # sell signals command
            if signal == "Sell":
                market_order_sl_price(currency, -1 * int(round(position_size * 0.75, 0)),
                                         round(data_ATR["c"][-1] + 0.00040, 5),
                                         round(data_ATR["c"][-1] - 0.00070, 5))
                print("New long position initiated for ", currency)
                print("sl = ", round(data_ATR["c"][-1] + 0.00040, 5))
                print("tp = ", round(data_ATR["c"][-1] - 0.00070, 5))

            if signal == "Strong_Sell":
                market_order_sl_price(currency, -1 * int(round(position_size * 1.2, 0)),
                                         round(data_ATR["c"][-1] + 0.00040, 5),
                                         round(data_ATR["c"][-1] - 0.00070, 5))
                print("New long position initiated for ", currency)
                print("sl = ", round(data_ATR["c"][-1] + 0.00040, 5))
                print("tp = ", round(data_ATR["c"][-1] - 0.00070, 5))

            print("##########################")

    except:
        print("error encountered....skipping this iteration")
        print("##########################")

    print("..........................")
    print("##### Next Iteration #####")
    print("..........................")

# main()


########### continuous execution ###########

start_time = time.time()
time_out = time.time() + 60*60*1
while time.time() <= time_out:
    try:
        print("passthrough at ",
              time.strftime('%Y-%m-%d %H:%M:%S',
                            time.localtime(time.time())))
        main()
        time.sleep(60.0 - ((time.time() - start_time) % 60.0))
    except KeyboardInterrupt:
        print('\n\nKeyboard exception received. Exiting.')
        exit()







        
        
        
    
    
    
    
    
    
    
    






