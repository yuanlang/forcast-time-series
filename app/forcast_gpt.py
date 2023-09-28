from sktime.datasets import load_airline
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import os
import json
import pandas as pd

# You have to generate your OpenAI API key first and assign it to
# an environmental variable
# For Linux, it looks as follows:
# export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
openai.api_key = "AK-cf18de71-37a6-46d0-ab04-acc187e66d10"
openai.api_base = "https://api.api2gpt.com/v1"


def chat_gpt_forecast(data, horizon,
                      time_idx='Period',
                      forecast_col='Forecast',
                      model="gpt-3.5-turbo",
                      verbose=False):

    prompt = f""" 
    Given the dataset delimited by the triple backticks, 
    forecast next {horizon} values of the time series. 

    Return the answer in JSON format, containing two keys: '{time_idx}' 
    and '{forecast_col}', and list of values assigned to them. 
    Return only the forecasts, not the Python code.

    ``` {data.to_string()}``` 
    """

    if verbose:
        print(prompt)

    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )

    output = response.choices[0].message["content"]

    try:
        json_object = json.loads(output)
        df = pd.DataFrame(json_object)
        df[time_idx] = df[time_idx].astype(data.index.dtype)
    except:
        df = output
        print(output)

    return df


if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = [20, 10]

    # Loading data
    air_passengers = load_airline()

    # Plotting
    sns.set_theme()
    # air_passengers.plot(title='Airline passengers 1949-1960')
    # plt.show()

    y_train = air_passengers[air_passengers.index < '1959-01']
    y_test = air_passengers[air_passengers.index >= '1959-01']

    horizon = 12
    y_train = air_passengers[air_passengers.index < '1959-01']
    y_test = air_passengers[air_passengers.index >= '1959-01']

    gpt_forecast = chat_gpt_forecast(y_train, horizon)
    y = air_passengers.reset_index()
    y.merge(gpt_forecast, how='outer') \
        .plot(x='Period', y=['Number of airline passengers', 'Forecast'])
    plt.show()
