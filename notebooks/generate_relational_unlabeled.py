# generate_relational_unlabeled.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

# ========================= AYARLAR =========================
RNG_SEED = 42
OUT_DIR = "."   # CSV'ler buraya yazılır

N_USERS    = 50_000
N_PRODUCTS = 5_000
N_ORDERS   = 300_000
N_PAYMENTS = N_ORDERS  # 1:1

DATE_START = "2023-01-01"
DATE_END   = "2025-09-01"

# --- Anomali oranları (ETİKET YOK; veriye gömülü) ---
P_BAD_FK_ORDER_USER         = 0.02   # orders.user_id geçersiz
P_BAD_FK_ORDER_PRODUCT      = 0.01   # orders.product_id geçersiz
P_NEGATIVE_OR_ABSURD_PRICE  = 0.01   # products.price negatif/çok büyük
P_RAPID_FIRE_USERS          = 0.01   # users'ın %1'i kısa sürede çok sipariş
RAPID_FIRE_ORDERS_PER_USER  = 150

P_PAYMENT_AMOUNT_MISMATCH   = 0.03   # payments.amount != orders.total_amount
P_PAYMENT_MISSING_METHOD    = 0.02   # payments.method eksik
P_PAYMENT_BAD_STATUS        = 0.01   # payments.status anlamsız
P_BAD_TIMESTAMP_ORDERS      = 0.003  # orders.order_date çok eski/gelecek
P_BAD_TIMESTAMP_PAYMENTS    = 0.003  # payments.payment_date çok eski/gelecek
P_DUPLICATE_ORDER_IDS       = 0.001  # orders.order_id tekrar
P_MISSING_VALUES_ORDERS     = 0.01   # orders'da NaN
P_MISSING_VALUES_USERS      = 0.01   # users'da NaN
P_MISSING_VALUES_PRODUCTS   = 0.01   # products'da NaN

rng = np.random.default_rng(RNG_SEED)

# ========================= YARDIMCI =========================
def random_datetimes(n, start=DATE_START, end=DATE_END):
    start = np.datetime64(start, "s")
    end   = np.datetime64(end, "s")
    span = int((end - start) / np.timedelta64(1, "s"))
    offs = rng.integers(0, span, size=n, endpoint=False)
    return start + offs.astype("timedelta64[s]")

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

# ========================= USERS =========================
def make_users(n=N_USERS):
    user_id = np.arange(1, n + 1, dtype=np.int64)

    age = rng.normal(35, 12, n).round().astype(float)
    age[age < 14] = rng.integers(14, 20, size=(age < 14).sum())  # child düzeltme
    bad_age_mask = rng.random(n) < 0.002  # saçma yaşlar
    age[bad_age_mask] = rng.choice([-5, -1, 130, 240], size=bad_age_mask.sum())

    cities = np.array(["Istanbul","Ankara","Izmir","Bursa","Antalya","Adana","Konya","Gaziantep","Unknown"])
    city = rng.choice(cities, size=n, p=[0.22,0.17,0.12,0.1,0.1,0.08,0.07,0.06,0.08])

    income = rng.lognormal(mean=8.8, sigma=0.6, size=n)  # sağa çarpık
    rich_mask = rng.random(n) < 0.005
    income[rich_mask] *= rng.uniform(5, 50, size=rich_mask.sum())

    signup_ts = random_datetimes(n)
    # Uç timestamp
    bt = rng.random(n) < 0.002
    signup_ts[bt] = rng.choice(np.array([np.datetime64("1985-01-01"), np.datetime64("2035-01-01")]), size=bt.sum())

    # NaN'ler
    miss_u = rng.random(n) < P_MISSING_VALUES_USERS
    age[miss_u] = np.nan
    miss_city = rng.random(n) < (P_MISSING_VALUES_USERS/2)
    if miss_city.any():
        city[miss_city] = None

    return pd.DataFrame({
        "user_id": user_id,
        "age": age,
        "city": city,
        "income": income,
        "signup_ts": signup_ts
    })

# ========================= PRODUCTS =========================
def make_products(n=N_PRODUCTS):
    product_id = np.arange(1, n + 1, dtype=np.int64)

    cats = np.array(["electronics","fashion","home","toys","sports","books","grocery","beauty"])
    category = rng.choice(cats, size=n, p=[0.18,0.15,0.14,0.08,0.12,0.1,0.15,0.08])

    base_price = rng.lognormal(mean=3.2, sigma=0.7, size=n).astype(float)
    pricey = rng.random(n) < 0.05
    base_price[pricey] *= rng.uniform(2, 6, size=pricey.sum())
    # negatif/absürt
    badp = rng.random(n) < P_NEGATIVE_OR_ABSURD_PRICE
    if badp.any():
        base_price[badp] = rng.choice([-99.0, -5.0, 9_999_999.0], size=badp.sum())

    # NaN
    miss_p = rng.random(n) < P_MISSING_VALUES_PRODUCTS
    if miss_p.any():
        category[miss_p] = None

    return pd.DataFrame({
        "product_id": product_id,
        "category": category,
        "price": base_price
    })

# ========================= ORDERS =========================
def make_orders(n=N_ORDERS, users_n=N_USERS, products_n=N_PRODUCTS):
    order_id   = np.arange(1, n + 1, dtype=np.int64)
    user_id    = rng.integers(1, users_n + 1, size=n, endpoint=True)
    product_id = rng.integers(1, products_n + 1, size=n, endpoint=True)

    # Geçersiz FK anomalileri
    badu = rng.random(n) < P_BAD_FK_ORDER_USER
    user_id[badu] = rng.integers(users_n + 1, users_n + 30_000, size=badu.sum(), endpoint=True)
    badp = rng.random(n) < P_BAD_FK_ORDER_PRODUCT
    product_id[badp] = rng.integers(products_n + 1, products_n + 20_000, size=badp.sum(), endpoint=True)

    order_date = random_datetimes(n)
    # >>> ÖNEMLİ: NaN enjekte edebilmek için float
    quantity = rng.integers(1, 6, size=n).astype(float)   # 1..5 fakat float
    total_amount = np.zeros(n, dtype=float)

    # Hızlı sipariş kullanıcıları (bot davranışı)
    n_rapid_users = max(1, int(N_USERS * P_RAPID_FIRE_USERS))
    rapid_user_ids = rng.choice(np.arange(1, users_n + 1), size=n_rapid_users, replace=False)
    for uid in rapid_user_ids:
        locs = np.where(user_id == uid)[0]
        if len(locs) == 0:
            continue
        take = min(RAPID_FIRE_ORDERS_PER_USER, len(locs))
        idx = rng.choice(locs, size=take, replace=False)
        base_time = rng.choice(order_date[idx])
        seconds = rng.integers(0, 3600, size=len(idx))
        order_date[idx] = base_time + seconds.astype("timedelta64[s]")

    # Uç timestamp
    bt = rng.random(n) < P_BAD_TIMESTAMP_ORDERS
    if bt.any():
        order_date[bt] = rng.choice(np.array([np.datetime64("1970-01-01"),
                                              np.datetime64("2039-01-19")]), size=bt.sum())

    # Duplicate order_id
    dup_mask = rng.random(n) < P_DUPLICATE_ORDER_IDS
    if dup_mask.any():
        order_id[dup_mask] = rng.integers(1, max(2, n//2), size=dup_mask.sum(), endpoint=True)

    # NaN değerler (özellikle quantity)
    miss_o = rng.random(n) < P_MISSING_VALUES_ORDERS
    quantity[miss_o] = np.nan

    return pd.DataFrame({
        "order_id": order_id,
        "user_id": user_id,
        "product_id": product_id,
        "order_date": order_date,
        "quantity": quantity,        # float (NaN tutar)
        "total_amount": total_amount # sonra dolduracağız
    })

# ========================= PAYMENTS =========================
def make_payments(n=N_PAYMENTS):
    payment_id = np.arange(1, n + 1, dtype=np.int64)
    methods = np.array(["card","cash","transfer","wallet"])
    method = rng.choice(methods, size=n, p=[0.55,0.1,0.25,0.1])
    status = rng.choice(np.array(["paid","pending","failed"]), size=n, p=[0.85,0.1,0.05])

    payment_date = random_datetimes(n)

    # Eksik method
    miss_method = rng.random(n) < P_PAYMENT_MISSING_METHOD
    if miss_method.any():
        method[miss_method] = None

    # Kötü status
    bad_status = rng.random(n) < P_PAYMENT_BAD_STATUS
    if bad_status.any():
        status[bad_status] = rng.choice(np.array(["unknown","void","???"]), size=bad_status.sum())

    # Uç timestamp
    bt = rng.random(n) < P_BAD_TIMESTAMP_PAYMENTS
    if bt.any():
        payment_date[bt] = rng.choice(np.array([np.datetime64("1969-12-31"),
                                                np.datetime64("2040-01-01")]), size=bt.sum())

    return pd.DataFrame({
        "payment_id": payment_id,
        "order_id": np.arange(1, n + 1, dtype=np.int64),  # 1:1 (bazısı uyumsuz olabilir)
        "method": method,
        "amount": np.zeros(n, dtype=float),               # sonra dolduracağız
        "status": status,
        "payment_date": payment_date
    })

# ========================= ÜRETİM / KAYDETME =========================
def main():
    ensure_dir(OUT_DIR)

    print("Users üretiliyor...")
    users = make_users()
    print("Products üretiliyor...")
    products = make_products()
    print("Orders üretiliyor...")
    orders = make_orders()
    print("Payments üretiliyor...")
    payments = make_payments()

    # total_amount hesapla (price * quantity)
    prod_price = products.set_index("product_id")["price"]
    q = orders["quantity"].fillna(1)  # NaN quantity → 1 varsay
    valid_price = prod_price.reindex(orders["product_id"]).fillna(0.0)
    orders["total_amount"] = (valid_price.values * q.values).round(2)

    # payments.amount = order_total (default)
    payments["amount"] = orders["total_amount"].reindex(payments["order_id"]).fillna(0.0).values

    # Payment uyumsuzluk anomalisi
    mm = rng.random(len(payments)) < P_PAYMENT_AMOUNT_MISMATCH
    if mm.any():
        sign = rng.choice([-1, 1], size=mm.sum())
        delta = rng.uniform(1, 200, size=mm.sum())
        newv = payments.loc[mm, "amount"].values + sign * delta
        payments.loc[mm, "amount"] = np.maximum(0.0, newv).round(2)

    # Order total’da uç değer anomalisi (az)
    weird_total_mask = rng.random(len(orders)) < 0.002
    if weird_total_mask.any():
        orders.loc[weird_total_mask, "total_amount"] = rng.choice(
            [-10.0, -99.0, 9_999_999.0], size=weird_total_mask.sum()
        )

    # --- CSV'leri yaz ---
    users_path    = os.path.join(OUT_DIR, "users.csv")
    products_path = os.path.join(OUT_DIR, "products.csv")
    orders_path   = os.path.join(OUT_DIR, "orders.csv")
    payments_path = os.path.join(OUT_DIR, "payments.csv")

    users.to_csv(users_path, index=False)
    products.to_csv(products_path, index=False)
    orders.to_csv(orders_path, index=False)
    payments.to_csv(payments_path, index=False)

    # --- ÖZET ---
    def nz(d): return {k:int(v) for k,v in d.items()}
    print("\n--- ÖZET ---")
    print("users:", users.shape, "NaN:", nz(users.isna().sum()))
    print("products:", products.shape, "NaN:", nz(products.isna().sum()))
    print("orders:", orders.shape, "NaN:", nz(orders.isna().sum()))
    print("payments:", payments.shape, "NaN:", nz(payments.isna().sum()))
    print("\nDosyalar yazıldı:")
    print(users_path)
    print(products_path)
    print(orders_path)
    print(payments_path)
    print("\nNotlar:")
    print("- Etiket YOK. Anomaliler veri içinde gömülü:")
    print("  * orders: geçersiz FK, duplicate order_id, NaN, uç timestamp, total_amount gariplikleri")
    print("  * products: negatif/aşırı fiyatlar, missing category")
    print("  * users: saçma yaşlar, uç timestamp, NaN")
    print("  * payments: amount != order total, missing method, bad status, uç timestamp")
    print("- 'rapid-fire' kullanıcılar: kısa sürede çok sipariş (davranışsal anomali)")
    print("- Amaç: join + kurallar veya ML ile anomalileri SEN tespit edeceksin.")

if __name__ == "__main__":
    main()
