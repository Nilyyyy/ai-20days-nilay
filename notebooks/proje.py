import duckdb

# DuckDB bağlantısı
con = duckdb.connect()

# CSV'yi tabloya yükle
con.execute(r"""
    CREATE TABLE http_dataset AS
    SELECT * FROM read_csv_auto('C:/Users/bunco/OneDrive/Masaüstü/http_dataset.csv', header=True);
""")

# Toplam satır sayısı
print("Toplam satır:", con.execute("SELECT COUNT(*) FROM http_dataset").fetchall())

# Normal / Anomali dağılımı
print(con.execute("SELECT label, COUNT(*) FROM http_dataset GROUP BY label").fetchall())

# İlk 5 satır
print(con.execute("SELECT * FROM http_dataset LIMIT 5").fetchdf())

# Bağlantıyı kapat
con.close()
