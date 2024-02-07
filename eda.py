import pandas as pd
import numpy as np
import plotly.express as px

sales = pd.read_csv('/content/train.csv')
store = pd.read_csv('/content/store.csv')

print(sales.head())
print(store.head())

df = sales.join(store.set_index('Store'), on='Store')
df['Date']= pd.to_datetime(df['Date'])
df['WeekDay'] = df['Date'].dt.weekday
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

print(df.head())

print('Braki danych w sales:')
for col in sales.columns:
  print(f'{col}: {df[col].isna().sum()}')
print('Braki danych w store:')
for col in store.columns:
  print(f'{col}: {df[col].isna().sum()}')

'''Do ustalenia:
1.   Czym są Promo, Promo2? (promocje? promowanie sklepu?)
2.   Jak potraktować braki danych? (występują tylko w 
danych store.csv - te dane nie są uszeregowane czasowo)
'''


sales_by_date = df.groupby(['Date'])['Sales'].mean()
fig = px.line(sales_by_date)
fig.show()

'''Widać sezonowość - wyższe wartości w okolicach Bożego Narodzenia.
W niedziele sklepy zamknięte'''

only_open = df[df['Open'] == 1]
sales_by_date_oo = only_open.groupby(['Date'])['Sales'].mean()
fig = px.line(sales_by_date_oo)
fig.show()

sales_by_weekday = df.groupby(['WeekDay'])['Sales'].mean()
fig = px.line(sales_by_weekday)
fig.show()

sales_by_day = df.groupby(['Day'])['Sales'].mean()
fig = px.line(sales_by_day)
fig.show()

''' Poniżej podobny wykres, ale tylko dla dni otwarcia sklepu. 
Wyraźniej widać, że sprzedaż wzrasta dwa razy w ciągu miesiąca - 
pod jego koniec oraz w samym środku miesiąca.'''

only_open = df[df['Open'] == 1]
sales_by_day_oo = only_open.groupby(['Day'])['Sales'].mean()
fig = px.line(sales_by_day_oo)
fig.show()

sales_by_month = df.groupby(['Month'])['Sales'].mean()
fig = px.line(sales_by_month)
fig.show()

'''Zdecydowany pik w grudniu (pewnie chodzi o święta), 
a najmniejsza sprzedaż w styczniu (może ludzie oszczędzają, 
bo wydali dużo na święta?)'''

sales_by_year = df.groupby(['Year'])['Sales'].mean()
fig = px.line(sales_by_year)
fig.show()

'''Wzrost sprzedaży na przestrzeni lat. Warto pamiętać, 
że średnia dla 2015 jest tylko z miesięcy do lipca włącznie - 
ale prawdopodobnie tendencja wzrostowa dla całego roku byłaby 
nadal zachowana i wzrost mógłby być nawet wyższy, bo średnia 
uwzględniałaby grudzień.'''

sales_by_storetype = df.groupby(['StoreType'])['Sales'].mean()
fig = px.bar(sales_by_storetype)
fig.show()

sales_by_assortment = df.groupby(['Assortment'])['Sales'].mean()
fig = px.bar(sales_by_assortment)
fig.show()

sales_by_promo = df.groupby(['Promo'])['Sales'].mean()
fig = px.bar(sales_by_promo)
fig.show()

sales_by_promo2 = df.groupby(['Promo2'])['Sales'].mean()
fig = px.bar(sales_by_promo2)
fig.show()

'''Różne wyniki dla Promo i Promo2 - wyższa sprzdaż przy 
obecności Promo, ale z drugiej strony niższa przy obecności Promo2. '''

sales_by_holiday = df.groupby(['StateHoliday'])['Sales'].mean()
fig = px.bar(sales_by_holiday)
fig.show()

sales_by_school = df.groupby(['SchoolHoliday'])['Sales'].mean()
fig = px.bar(sales_by_school)
fig.show()

store_mean = df.groupby(['Store'])['Sales', 'CompetitionDistance'].mean()
fig = px.scatter(x=store_mean['CompetitionDistance'], y=store_mean['Sales'], trendline="ols")
fig.show()
