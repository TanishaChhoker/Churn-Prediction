import streamlit as st
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.ticker as mtick  
import matplotlib.pyplot as plt

def classify():
    telco_base_data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Function for generating bar plot
    def generate_plot():
        telco_base_data['Churn'].value_counts().plot(kind='barh', figsize=(8, 6))
        plt.xlabel("Count", labelpad=14)
        plt.ylabel("Target Variable", labelpad=14)
        plt.title("Count of TARGET Variable per category", y=1.02)
        return plt
    
    fig = generate_plot()
    
    # Display bar plot
    st.pyplot(fig)

    

    telco_data = telco_base_data.copy()

    telco_data.TotalCharges = pd.to_numeric(telco_data.TotalCharges, errors='coerce')


    telco_data.dropna(how = 'any', inplace = True)


    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]

    telco_data['tenure_group'] = pd.cut(telco_data.tenure, range(1, 80, 12), right=False, labels=labels)



    telco_data.drop(columns= ['customerID','tenure'], axis=1, inplace=True)


    for i, predictor in enumerate(telco_data.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
        plt.figure(i)
        sns.countplot(data=telco_data, x=predictor, hue='Churn')

    st.pyplot(plt)

    
    telco_data['Churn'] = np.where(telco_data.Churn == 'Yes',1,0)

    telco_data_dummies = pd.get_dummies(telco_data)


    sns.lmplot(data=telco_data_dummies, x='MonthlyCharges', y='TotalCharges', fit_reg=False)

    st.pyplot(plt)


    plt.figure(figsize=(20,8))
    telco_data_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')
    st.pyplot(plt)

    plt.figure(figsize=(12,12))
    sns.heatmap(telco_data_dummies.corr(), cmap="Paired")
    st.pyplot(plt)

    

    new_df1_target0=telco_data.loc[telco_data["Churn"]==0]
    new_df1_target1=telco_data.loc[telco_data["Churn"]==1]

    def uniplot(df,col,title,hue =None):

        sns.set_style('whitegrid')
        sns.set_context('talk')
        plt.rcParams["axes.labelsize"] = 20
        plt.rcParams['axes.titlesize'] = 22
        plt.rcParams['axes.titlepad'] = 30


        temp = pd.Series(data = hue)
        fig, ax = plt.subplots()
        width = len(df[col].unique()) + 7 + 4*len(temp.unique())
        fig.set_size_inches(width , 8)
        plt.xticks(rotation=45)
        plt.yscale('log')
        plt.title(title)
        ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue,palette='bright')

        return plt
    
    uniplot(new_df1_target1,col='Partner',title='Distribution of Gender for Churned Customers',hue='gender')
    st.pyplot(plt)
    uniplot(new_df1_target0,col='Partner',title='Distribution of Gender for Non Churned Customers',hue='gender')
    st.pyplot(plt)
    uniplot(new_df1_target1,col='PaymentMethod',title='Distribution of PaymentMethod for Churned Customers',hue='gender')
    st.pyplot(plt)
    uniplot(new_df1_target1,col='Contract',title='Distribution of Contract for Churned Customers',hue='gender')
    st.pyplot(plt)
    uniplot(new_df1_target1,col='TechSupport',title='Distribution of TechSupport for Churned Customers',hue='gender')
    st.pyplot(plt)
    uniplot(new_df1_target1,col='SeniorCitizen',title='Distribution of SeniorCitizen for Churned Customers',hue='gender')
    st.pyplot(plt)

    return None


st.title('Churn Analysis - EDA')

if st.button('Start'):
    classify()
