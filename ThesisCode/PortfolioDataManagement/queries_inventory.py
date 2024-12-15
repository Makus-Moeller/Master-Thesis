import sqlalchemy as sa
import pandas as pd

def fetch_transactions_current_day_Isin(conn, formatted_prev_date, formatted_trade_time, instrument_id, portfolio):
    query = sa.text(f'''select sum(Number) as currentDaySum
            from [PR_DeltaDB_JB].[dbo].[Transactions]  
            where GETDATE() < Deleted
            and Portfolio = :portfolio
            and InstrumentID = :InstrumentID
            and :formatted_prev_day < Settlement
            and TradeTime <= :formatted_trade_time 
            ''')
    
    # Execute the query with the trade_time as a parameter
    result_df = pd.read_sql_query(query, conn, params={"formatted_prev_day": formatted_prev_date, 
                                                    "formatted_trade_time": formatted_trade_time, 
                                                    "InstrumentID": instrument_id,
                                                    "portfolio": portfolio})
    
    value = result_df['currentDaySum'].iloc[0] 
    return value if value is not None else 0


def fetch_inventory_prev_day_rows_and_currency(conn, formatted_prev_date, portfolio):
    query = sa.text(f'''
        SELECT distinct P.InstrumentID, P.Number, IRD.CurrencyID
        FROM [PR_DeltaDB_JB].[dbo].[Position] P
            JOIN [PR_DeltaDB].[dbo].[InstrumentRepositoryData] IRD
            ON P.InstrumentID COLLATE SQL_Latin1_General_CP1_CI_AS = IRD.InstrumentID COLLATE SQL_Latin1_General_CP1_CI_AS
        WHERE GETDATE() < P.Deleted
            AND P.PositionRule = :portfolio
            AND P.ValidFrom = :formatted_prev_date
            ORDER BY P.InstrumentID;
    ''')
    
    # Execute the query
    result = pd.read_sql_query(query, conn, params={"formatted_prev_date": formatted_prev_date, "portfolio": portfolio})
    return result


def fetch_bpv_bulk(conn, instrument_ids, trade_time):
    # Format the list of instrument IDs to be used in the SQL query
    instrument_ids_placeholder = ', '.join([f"'{isin}'" for isin in instrument_ids])
    
    query_bpv = sa.text(f'''
        WITH RankedBPV AS (
            SELECT InstrumentID, Number AS BPV, 
                   ROW_NUMBER() OVER (PARTITION BY InstrumentID ORDER BY ValidFrom DESC) AS rn
            FROM [PR_DeltaDB].[dbo].[KeyRatio]
            WHERE GETDATE() < Deleted
              AND Context = 'Risikostyring'
              AND KeyRatioName = 'BPV'
              AND ValidFrom < :trade_time
              AND InstrumentID IN ({instrument_ids_placeholder})
        )
        SELECT InstrumentID, BPV
        FROM RankedBPV
        WHERE rn = 1
        order by InstrumentID
    ''')

    # Execute the query and return the result as a dictionary
    result_df = pd.read_sql_query(query_bpv, conn, params={'trade_time': trade_time})
    return (result_df.set_index('InstrumentID')['BPV']/100).to_dict()

def fetch_collateral_prev_day_bulk(conn, instrument_ids, formatted_prev_date):
    # Format the list of instrument IDs to be used in the SQL query
    instrument_ids_placeholder = ', '.join([f"'{isin}'" for isin in instrument_ids])
    
    query = sa.text(f'''
        SELECT InstrumentID, sum(Number) as collateralSum
        FROM [PR_DeltaDB_JB].[dbo].[Position]
        WHERE GETDATE() < Deleted
          AND PositionRule IN ('JMS07', 'JMS08', 'JMS14', 'JMS17', 'JMS18')
          AND ValidFrom = :formatted_prev_day
          AND InstrumentID IN ({instrument_ids_placeholder})
        GROUP BY InstrumentID
    ''')

    result_df = pd.read_sql_query(query, conn, params={'formatted_prev_day': formatted_prev_date})
    return result_df.set_index('InstrumentID')['collateralSum'].to_dict()


def fetch_transactions_current_day_bulk(conn, instrument_ids, formatted_prev_date, formatted_trade_time, portfolio):
    # Format the list of instrument IDs to be used in the SQL query
    instrument_ids_placeholder = ', '.join([f"'{isin}'" for isin in instrument_ids])
    
    
    query = sa.text(f'''
        SELECT InstrumentID, sum(Number) as currentDaySum
        FROM [PR_DeltaDB_JB].[dbo].[Transactions]
        WHERE GETDATE() < Deleted
          AND Portfolio = :portfolio
          AND InstrumentID IN ({instrument_ids_placeholder})
          AND :formatted_prev_day < Settlement
          AND TradeTime <= :formatted_trade_time
        GROUP BY InstrumentID
    ''')

    result_df = pd.read_sql_query(query, conn, params={'formatted_prev_day': formatted_prev_date, 
                                                       'formatted_trade_time': formatted_trade_time,
                                                       'portfolio': portfolio })
    
    return result_df.set_index('InstrumentID')['currentDaySum'].to_dict()

def fetch_collateral_current_day_bulk(conn, instrument_ids, formatted_prev_day, formatted_trade_time):
    # Format the list of instrument IDs to be used in the SQL query
    instrument_ids_placeholder = ', '.join([f"'{isin}'" for isin in instrument_ids])
    
    query = sa.text(f'''
        SELECT InstrumentID, sum(Number) as currentDayCollateralSum
        FROM [PR_DeltaDB_JB].[dbo].[Transactions]
        WHERE GETDATE() < Deleted
          AND Portfolio IN ('JMS07', 'JMS08', 'JMS14', 'JMS17', 'JMS18')
          AND InstrumentID IN ({instrument_ids_placeholder})
          AND :formatted_prev_day < Settlement
          AND TradeTime <= :formatted_trade_time
        GROUP BY InstrumentID
    ''')

    result_df = pd.read_sql_query(query, conn, params={'formatted_prev_day': formatted_prev_day, 
                                                       'formatted_trade_time': formatted_trade_time})
    return result_df.set_index('InstrumentID')['currentDayCollateralSum'].to_dict()

def fetch_new_positions(conn, formatted_prev_date, formatted_trade_time, portfolio):
    query = sa.text(f'''(SELECT InstrumentID
                FROM [PR_DeltaDB_JB].[dbo].[Transactions]
                WHERE GETDATE() < Deleted
                AND Portfolio = :portfolio
                AND :formatted_prev_date < Settlement
                AND TradeTime <= :formatted_trade_time)

                EXCEPT

                (SELECT InstrumentID
                FROM [PR_DeltaDB_JB].[dbo].[Position]
                WHERE GETDATE() < Deleted
                AND PositionRule = :portfolio 
                AND ValidFrom = :formatted_prev_date)''')
    
    result_df = pd.read_sql_query(query, conn, params={"formatted_prev_date": formatted_prev_date, 
                                                    "formatted_trade_time": formatted_trade_time,
                                                    "portfolio": portfolio})
    return result_df


def fetch_collateral_current_day(conn, formatted_prev_day, formatted_trade_time, instrument_id):
    query = sa.text(f'''select sum(Number) as currentDayCollateralSum
        from [PR_DeltaDB_JB].[dbo].[Transactions]  
        where GETDATE() < Deleted
        and Portfolio in ('JMS07',
						'JMS08',
						'JMS14',
						'JMS17',
						'JMS18'
						)
        and InstrumentID = :InstrumentID
        and :formatted_prev_day < Settlement
        and TradeTime <= :formatted_trade_time 
            ''')
    
    # Execute the query with the trade_time as a parameter
    result_df = pd.read_sql_query(query, conn, params={"formatted_prev_day": formatted_prev_day, 
                                                    "formatted_trade_time": formatted_trade_time, 
                                                    "InstrumentID": instrument_id})
    
    value = result_df['currentDayCollateralSum'].iloc[0] 
    return value if value is not None else 0


def fetch_currency(conn, instrument_id):
    query = sa.text(f'''select distinct CurrencyID 
            from InstrumentRepositoryData
            where InstrumentID = :InstrumentID
            ''')
    result_df = pd.read_sql_query(query, conn, params={ "InstrumentID": instrument_id})
    
    currency = result_df['CurrencyID'].iloc[0] 
    assert currency is not None, "Currency not retrieved"
        
    return currency 


# Function to fetch BPV for a given InstrumentID and TradeTime
def fetch_bpv(conn, instrument_id, trade_time):
    query_bpv = sa.text('''
        SELECT TOP 1 [Number] AS BPV
        FROM [PR_DeltaDB].[dbo].[KeyRatio]
        WHERE GETDATE() < Deleted
          AND InstrumentID = :instrument_id
          AND Context = 'Risikostyring'
          AND KeyRatioName = 'BPV'
          AND ValidFrom < :trade_time
        ORDER BY ValidFrom DESC
    ''')

    result_bpv = pd.read_sql_query(query_bpv, conn, params={'instrument_id': instrument_id, 'trade_time': trade_time})

    # If there is a result, return the BPV, otherwise return 0.0 or NaN
    return result_bpv['BPV'].iloc[0] / 100 if not result_bpv.empty else 0.0