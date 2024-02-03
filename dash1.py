import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import statsmodels.formula.api as smf

def load_data():
    data = pd.read_csv('df2.csv')
    return data
st.header("💎 Diamond price analysys 💎 ")
df = load_data()
df = df.drop(columns = 'Unnamed: 0')
plt.style.use('default')
#plt.style.use('dark_background')
col1, col2 = st.columns(2)
st.set_option('deprecation.showPyplotGlobalUse', False)

with col1:
    st.header("Wybierz zmienną")
    variables = df.columns.tolist()
    selected_variable = st.selectbox('Wybierz zmienną', variables)

with col2:
    st.header("Podgląd danych")
    if selected_variable:
        if selected_variable != 'price':
            # Wyświetlanie wybranej zmiennej i 'price', gdy są różne
            st.write(df[[selected_variable, 'price']].head())
        else:
            # Wyświetlanie tylko 'price', gdy wybrana zmienna to 'price'
            st.write(df['price'].head())



st.header("Wykres")
if selected_variable:
    if selected_variable == 'price':
        st.write(f"Histogram i wykres gęstości dla {selected_variable}")
        sns.histplot(df[selected_variable], kde=True)
        st.pyplot()
    elif df[selected_variable].dtype == 'object':
        st.write(f"Boxplot dla {selected_variable} vs cena")
        sns.boxplot(data=df, x=selected_variable, y='price', palette = "husl")
        st.pyplot()
    else:
        st.write(f"Scatterplot dla {selected_variable} vs cena")
        sns.scatterplot(data=df, x=selected_variable, y='price')
        st.pyplot()

#załadowanie datasetu ze zmodyfikowanymi zmiennymi kategorycznymi
def load_reg_data():
    data = pd.read_csv('df3.csv')
    return data

df_reg = load_data()
st.header("Regresja 📈")
#zmienne reg_var wybrane na podstawie modelowania zawartego w notatniku data_cleaing_and_regression.ipynb
reg_var = ['carat', 'x_dimension', 'clarity', 'cut']
selected_regression = st.selectbox('Wybierz zmienną', reg_var)
if selected_regression:
    if selected_regression == 'carat':
        model1 = smf.ols(formula="price ~ carat", data=df_reg).fit()
        st.write(model1.summary())
        df_reg["carat_fitted"] = model1.fittedvalues
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_reg['carat'], y=df_reg['price'], name="Carat vs Price", mode="markers"))
        fig.add_trace(go.Scatter(
            x=df_reg["carat"], y=df_reg["carat_fitted"], name="Model regresji"))
        fig.update_layout(title="Linia regresji carat vs price", xaxis_title="carat",
            yaxis_title="price")
        st.plotly_chart(fig)
    
    elif selected_regression == 'x_dimension':
        model2 = smf.ols(formula="price ~ x_dimension", data=df_reg).fit()
        st.write(model2.summary())
        df_reg["x_dimension_fitted"] = model2.fittedvalues
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_reg['x_dimension'], y=df_reg['price'], name="x_dimension vs Price", mode="markers"))
        fig.add_trace(go.Scatter(
            x=df_reg["x_dimension"], y=df_reg["x_dimension_fitted"], name="Model regresji"))
        fig.update_layout(title="Linia regresji x_dimension vs price", xaxis_title="x_dimension",
            yaxis_title="price")
        st.plotly_chart(fig)

    elif selected_regression == 'clarity':
        model3 = smf.ols(formula="price ~ clarity_encoded", data=df_reg).fit()
        st.write(model3.summary())
        df_reg["clarity_encoded_fitted"] = model3.fittedvalues
        # sns.set(rc = {'figure.figsize':(16,9)})
        # sns.regplot(x="clarity_encoded", y="price", data=df_reg, line_kws=dict(color="r"))
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df_reg['clarity_encoded'], y=df_reg['price'], label="Dane", marker='o') 
        sns.lineplot(x=df_reg['clarity_encoded'], y=df_reg['clarity_encoded_fitted'], color='red', label="Linia regresji")
        plt.title("Linia regresji dla clarity_encoded vs price")
        plt.xlabel("clarity_encoded")
        plt.ylabel("price")
        st.pyplot()
    
    elif selected_regression == 'cut':
        model4 = smf.ols(formula="price ~ cut_encoded", data=df_reg).fit()
        st.write(model4.summary())
        df_reg["cut_encoded_fitted"] = model4.fittedvalues
        # sns.set(rc = {'figure.figsize':(16,9)})
        # sns.regplot(x="cut_encoded", y="price", data=df_reg, line_kws=dict(color="r"))
        # st.pyplot()
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df_reg['cut_encoded'], y=df_reg['price'], label="Dane", marker='o') 
        sns.lineplot(x=df_reg['cut_encoded'], y=df_reg['cut_encoded_fitted'], color='red', label="Linia regresji")
        plt.title("Linia regresji dla cut_encoded vs price")
        plt.xlabel("cut_encoded")
        plt.ylabel("price")
        st.pyplot()
