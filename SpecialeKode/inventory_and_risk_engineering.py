from sqlalchemy.engine import URL
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from queries_inventory import *
import pandas

def create_connections():
    #make connection to DeltaDB
    connection_string = '''Driver={SQL Server Native Client 11.0}; 
                            Server=SQL-PR_DeltaDB_RFQ-PR,1521; 
                            Database=PR_DeltaDB;
                            Trusted_Connection=yes;'''

    connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

    engine_DeltaDB = create_engine(connection_url)

    #make connection to DeltaDB
    connection_string = '''Driver={SQL Server Native Client 11.0}; 
                            Server=SQL-PR_DeltaDB_RFQ-PR,1521; 
                            Database=PR_DeltaDB_JB;
                            Trusted_Connection=yes;'''

    connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

    engine_DeltaDB_JB = create_engine(connection_url)

    return engine_DeltaDB, engine_DeltaDB_JB


def get_inventory_previous_date(trade_time, conn, portfolio):
    # Parse the dates into a datetime objects and strings
    date_object = datetime.strptime(trade_time, '%m/%d/%Y %H:%M:%S')

    trade_time_formatted = date_object.strftime('%Y-%m-%d %H:%M:%S') 

    prev_day = date_object - timedelta(days=1)

    # Convert the datetime object to the desired format
    formatted_prev_date = prev_day.strftime('%Y-%m-%d')

    # Initialize the result variable
    inventory_previous_day = fetch_inventory_prev_day_rows_and_currency(conn, formatted_prev_date, portfolio)
    
    # While loop to keep subtracting days until a valid result is found
    while inventory_previous_day.empty:
        # Subtract one more day
        prev_day -= timedelta(days=1)
        formatted_prev_date = prev_day.strftime('%Y-%m-%d')

        # Fetch inventory data again for the updated previous day
        inventory_previous_day = fetch_inventory_prev_day_rows_and_currency(conn, formatted_prev_date, portfolio)
    
    return inventory_previous_day, trade_time_formatted, formatted_prev_date

def add_inventory_and_risk(df):

    length = len(df)
    engine_DeltaDB, engine_DeltaDB_JB = create_connections()

    # Open a connection using the SQLAlchemy engine
    with engine_DeltaDB_JB.begin() as conn_DB_JB, engine_DeltaDB.begin() as conn_DB:
            
            # Iterate over the rows in the bond trades dataframe
            for index, row in df.iterrows():
                trade_time = row['TradeTime']
                portfolio = 'jms04' if row['BookName'] == 'KORTMM' else 'jms05'
                
                #get inventory previous day
                inventory_previous_day, trade_time_formatted, formatted_prev_date = get_inventory_previous_date(trade_time, conn_DB_JB, portfolio)

                instrument_ids = inventory_previous_day.InstrumentID.to_list()
                
                #fetch necessary information for all Isins
                bpv_dict = fetch_bpv_bulk(conn_DB, instrument_ids, trade_time_formatted)
                collateral_prev_day_dict = fetch_collateral_prev_day_bulk(conn_DB_JB, instrument_ids, formatted_prev_date)
                inventory_curr_day_dict = fetch_transactions_current_day_bulk(conn_DB_JB, instrument_ids, formatted_prev_date, trade_time_formatted, portfolio)
                collateral_curr_day_dict = fetch_collateral_current_day_bulk(conn_DB_JB, instrument_ids, formatted_prev_date, trade_time_formatted)

                #Initialize values
                inventory = 0
                bpv_risk = 0
                
                for _, row in inventory_previous_day.iterrows():
                    isin = row['InstrumentID']
                    inventory_prev_day = row['Number']
                    currency = row['CurrencyID']  # This could also potentially be batched
                    currency_multiplier = 7.45 if currency == 'EUR' else 1.0
                    
                    collateral_prev_day = collateral_prev_day_dict.get(isin, 0.0)
                    inventory_curr_day = inventory_curr_day_dict.get(isin, 0.0)
                    collateral_curr_day = collateral_curr_day_dict.get(isin, 0.0)
                    bpv = bpv_dict.get(isin, 0.0) 

                    # Process risk and inventory calculations
                    risk_inventory_isin = (inventory_prev_day + collateral_prev_day + inventory_curr_day + collateral_curr_day) * currency_multiplier
                    inventory += risk_inventory_isin
                    bpv_risk += risk_inventory_isin*bpv
    
                # Add transactions current day for isin not found in Position table 
                new_positions = fetch_new_positions(conn_DB_JB, formatted_prev_date, trade_time_formatted, portfolio)
                
                
                isin_list_new = new_positions['InstrumentID'].to_list()

                if isin_list_new:                
                    inventory_curr_day_dict = fetch_transactions_current_day_bulk(conn_DB_JB, isin_list_new, formatted_prev_date, trade_time_formatted, portfolio)
                    collateral_curr_day_dict = fetch_collateral_current_day_bulk(conn_DB_JB, isin_list_new, formatted_prev_date, trade_time_formatted)
                    bpv_dict = fetch_bpv_bulk(conn_DB, isin_list_new, trade_time_formatted)
                    
                    for isin in isin_list_new:
                        
                        inventory_curr_day = inventory_curr_day_dict.get(isin, 0.0)
                        collateral_curr_day = collateral_curr_day_dict.get(isin, 0.0)
                        bpv = bpv_dict.get(isin, 0.0)
                        currency = fetch_currency(conn_DB, isin)
                        currency_multiplier = 7.45 if currency == 'EUR' else 1.0
                        
                        risk_inventory_isin = (inventory_curr_day+collateral_curr_day)*currency_multiplier
                        inventory += risk_inventory_isin
                        bpv_risk += risk_inventory_isin*bpv
                        
                #Append values to original df
                df.at[index, 'inventoryRisk'] = (inventory)/1000
                df.at[index, 'bpv_risk'] = (bpv_risk)/1000
                
                #print progress
                if (index+1) % 100 == 0:
                     print(f'Pogress: {index+1}/{length}') 
    
    df.to_csv("data/data_inventory_risk.csv")            
    
    return df