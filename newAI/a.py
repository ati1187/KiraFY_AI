import pandas as pd

def temizle_veri(dosya_yolu: str) -> pd.DataFrame:
    """
    Ham veriyi okuyup temiz ve dönüştürülmüş DataFrame olarak döner.
    """
    df = pd.read_csv(dosya_yolu, encoding='utf-8-sig')

    # Gereksiz sütunları çıkar
    if 'address' in df.columns:
        df.drop('address', axis=1, inplace=True)

    # Net m2'den " m2" ifadesini kaldır ve float'a çevir
    if 'NetSquareMeters' in df.columns:
        df['NetSquareMeters'] = df['NetSquareMeters'].astype(str).str.replace(' m2', '', regex=False)
        df['NetSquareMeters'] = pd.to_numeric(df['NetSquareMeters'], errors='coerce')

    # BuildingAge ordinal map
    age_map = {
        "0": 0,
        "1-5": 1,
        "5-10": 2,
        "11-15": 3,
        "16-20": 4,
        "21 Ve Üzeri": 5
    }
    df['BuildingAge'] = df['BuildingAge'].map(age_map)

    # NumberOfRooms: "3+1" → 3
    df['NumberOfRooms'] = df['NumberOfRooms'].str.extract(r'(\d+)').astype(float)

    # Sayısal olması gereken kolonlar
    sayisal_kolonlar = [
        'GrossSquareMeters', 'NetSquareMeters', 'NumberOfBathrooms',
        'NumberOfWCs', 'NumberFloorsofBuilding', 'NumberOfRooms'
    ]
    for kolon in sayisal_kolonlar:
        df[kolon] = pd.to_numeric(df[kolon], errors='coerce')

    # Eksik veri silme
    df = df.dropna()

    # Kategorik değişkenleri one-hot encode et
    kategorik_kolonlar = [
        'district', 'FloorLocation', 'HeatingType', 'CreditEligibility',
        'InsideTheSite', 'MortgageStatus', 'Swap', 'Balcony',
        'PriceStatus', 'ItemStatus', 'BuildStatus'
    ]
    df = pd.get_dummies(df, columns=kategorik_kolonlar)

    return df

# Fonksiyonu çalıştır ve sonucu göster
if __name__ == "__main__":
    temiz_df = temizle_veri("HouseData.csv")
    print("🔍 Temizlenmiş veri örneği:")
    print(temiz_df.head())
