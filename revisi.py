import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from mlxtend.frequent_patterns import association_rules, apriori

# load dataset
df = pd.read_excel("WarungFlamboyan.xlsx")
df["Waktu"] = pd.to_datetime(df["Waktu"], format="%d-%m-%Y %H:%M")

df["month"] = df["Waktu"].dt.month

df["month"].replace(
    [i for i in range(1, 12 + 1)],
    [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ],
    inplace=True,
)

page = st.sidebar.selectbox("Choose a page", ["Persebaran Data", "Apriori"])

if page == "Persebaran Data":
    st.title("Persebaran Data")

    # Menampilkan pie chart dari seluruh data barang
    all_barang_count_total = df["Barang"].value_counts()

    # Buat DataFrame baru yang berisi frekuensi setiap barang
    df_frekuensi_barang = pd.DataFrame(
        {
            "Barang": all_barang_count_total.index,
            "Frekuensi": all_barang_count_total.values,
        }
    )

    fig_pie_chart = px.pie(
        df_frekuensi_barang,
        names="Barang",
        values="Frekuensi",
        title="Persentase Jumlah Transaksi Tiap Barang",
    )

    st.plotly_chart(fig_pie_chart)

    # Membuat bar chart dari frekuensi barang terjual tiap bulan
    fig_bar_chart_bulanan = px.bar(
        df.groupby("month")["Barang"].count().reset_index(name="Frekuensi"),
        x="month",
        y="Frekuensi",
        title="Jumlah Transaksi Barang Tiap Bulan",
        labels={"month": "Bulan", "Frekuensi": "Jumlah Transaksi"},
        category_orders={
            "month": [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]
        },
    )
    st.plotly_chart(fig_bar_chart_bulanan)

elif page == "Apriori":
    st.title(
        "Market Basket Analysis Penjualan Jajan di Warung Flamboyan Menggunakan Algoritma Apriori"
    )

    def get_data():
        return df.copy()

    def preprocess_data(data):
        # Hapus data dengan satu transaksi dan satu barang saja
        data_filtered = data.groupby("ID")["Barang"].nunique().reset_index()
        data_filtered = data_filtered[data_filtered["Barang"] > 1]
        return data[data["ID"].isin(data_filtered["ID"])]

    def user_input_features():
        barang = st.selectbox(
            "Barang",
            [
                "Ahh",
                "Beng beng",
                "Chocolatos",
                "Chocopie",
                "Choki choki",
                "Gery salut",
                "Kimbo",
                "Momogi",
                "Nextar",
                "Oreo",
                "Sari gandum",
                "Shor",
                "Slai olai",
                "Superstar",
                "Yupi",
            ],
        )
        min_support = st.slider("Minimum Support", 0.01, 0.3, 0.05, 0.01)
        min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.5, 0.1)
        return barang, min_support, min_confidence

    barang, min_support, min_confidence = user_input_features()

    data = get_data()
    data = preprocess_data(data)

    def encode(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1

    if type(data) != type("No Result!"):
        barang_count = (
            data.groupby(["ID", "Barang"])["Barang"].count().reset_index(name="Count")
        )
        barang_count_pivot = barang_count.pivot_table(
            index="ID", columns="Barang", values="Count", aggfunc="sum"
        ).fillna(0)
        barang_count_pivot = barang_count_pivot.map(encode)

        frequent_barangs = apriori(
            barang_count_pivot, min_support=min_support, use_colnames=True
        )

        metric = "lift"
        min_threshold = 1

        rules = association_rules(
            frequent_barangs, metric=metric, min_threshold=min_threshold
        )[["antecedents", "consequents", "support", "confidence", "lift"]]
        rules.sort_values("confidence", ascending=False, inplace=True)

        def parse_list(x):
            x = list(x)
            if len(x) == 1:
                return x[0]
            elif len(x) > 1:
                return ", ".join(x)

        def return_barang_df(barang_antecedents):
            data = rules[
                ["antecedents", "consequents", "support", "confidence", "lift"]
            ].copy()

            data = data[data["confidence"] >= min_confidence]

            data["antecedents"] = data["antecedents"].apply(parse_list)
            data["consequents"] = data["consequents"].apply(parse_list)

            st.write(data)

            filtered_data = data.loc[data["antecedents"] == barang_antecedents]
            if not filtered_data.empty:
                return list(filtered_data.iloc[0, :])
            else:
                return None

        st.markdown("Hasil Rule Apriori : ")

        if not rules.empty:
            result = return_barang_df(barang)
            if result is not None:
                st.success(
                    f"Jika konsumen membeli **{barang}**, maka membeli **{result[1]}** secara bersamaan."
                )
            else:
                st.warning(f"Tidak ada rule untuk **{barang}**.")
        else:
            st.error("Tidak ada rule yang dapat ditampilkan.")

        st.markdown("Rekomendasi : ")

        if not rules.empty:
            if result is not None:
                st.success(
                    f"Letakkan **{barang}**, berdampingan dengan **{result[1]}**."
                )
            else:
                st.warning(
                    f"Tidak ada rekomendasi peletakan barang untuk **{barang}**."
                )
        else:
            st.error("Tidak ada rekomendasi yang dapat ditampilkan.")
