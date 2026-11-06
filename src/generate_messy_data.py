import numpy as np, pandas as pd, random
rng = np.random.default_rng(42); random.seed(42)
N = 300_000

cities = {"Karachi":.30,"Lahore":.22,"Islamabad":.12,"Rawalpindi":.08,"Faisalabad":.07,
          "Multan":.05,"Peshawar":.05,"Quetta":.03,"Hyderabad":.04,"Sialkot":.04}
city_vars = {"Karachi":["Karachi","karachi","KARACHI","Karchi","Khi "],
 "Lahore":["Lahore","lahore","LAHORE","Lahor","Lhr"],
 "Islamabad":["Islamabad","islamabad","ISLAMABAD","Islamabd"," Islamabad"],
 "Rawalpindi":["Rawalpindi","rawalpindi","Pindi","RAWALPINDI"],
 "Faisalabad":["Faisalabad","faisalabad","Faislabad"],
 "Multan":["Multan","multan","MULTAN"],"Peshawar":["Peshawar","peshawar","Peshawr"],
 "Quetta":["Quetta","quetta","QUETTA"],"Hyderabad":["Hyderabad","hyderabad","Hyd"],
 "Sialkot":["Sialkot","sialkot","SIALKOT"]}
products = {
 "Electronics":[("Wireless Earbuds",4500),("Power Bank 10000mAh",3200),("Smart Watch",8500),("Bluetooth Speaker",5200),("USB-C Cable",650)],
 "Fashion":[("Lawn Suit 3pc",6500),("Men Kurta",3800),("Sneakers",7200),("Handbag",5400),("Sunglasses",1800)],
 "Home":[("Bedsheet Set",4200),("Air Fryer",14500),("Dinner Set",9800),("Wall Clock",2100),("Blender",6800)],
 "Beauty":[("Face Wash",750),("Vitamin C Serum",1800),("Perfume 100ml",4500),("Sunblock SPF50",1500),("Hair Oil",600)],
 "Grocery":[("Basmati Rice 5kg",2100),("Cooking Oil 5L",3400),("Tea 900g",1700),("Dates 1kg",1500),("Honey 500g",1900)]}
flat = [(c,p,pr) for c,l in products.items() for p,pr in l]
first = ["Ali","Ahmed","Sara","Fatima","Usman","Ayesha","Hamza","Zainab","Bilal","Hira","Omar","Maryam","Hassan","Noor","Imran","Sana"]
last = ["Khan","Malik","Sheikh","Ansari","Qureshi","Siddiqui","Baig","Raza","Butt","Chaudhry"]

n_cust = 60_000
cust_first = rng.choice(first, n_cust); cust_last = rng.choice(last, n_cust)
cust_city = rng.choice(list(cities), n_cust, p=list(cities.values()))
cust_phone = ["03%02d%07d" % (rng.integers(0,50), rng.integers(0,10**7)) for _ in range(n_cust)]

ci = rng.integers(0, n_cust, N)
pi = rng.integers(0, len(flat), N)
dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, N), unit="D")
df = pd.DataFrame({
 "order_id": np.arange(100000, 100000+N),
 "customer_id": ["C%05d" % i for i in ci],
 "customer_name": [f"{cust_first[i]} {cust_last[i]}" for i in ci],
 "email": [f"{cust_first[i].lower()}.{cust_last[i].lower()}{i}@{random.choice(['gmail.com','yahoo.com','hotmail.com','outlook.com'])}" for i in ci],
 "phone": [cust_phone[i] for i in ci],
 "city": [cust_city[i] for i in ci],
 "order_date": dates,
 "category": [flat[i][0] for i in pi],
 "product": [flat[i][1] for i in pi],
 "quantity": rng.choice([1,1,1,2,2,3,4,5], N),
 "unit_price": [flat[i][2] for i in pi],
 "discount": rng.choice([0,0,0,0.05,0.1,0.15,0.2,0.3], N),
 "payment_method": rng.choice(["COD","Card","JazzCash","Easypaisa","Bank Transfer"], N, p=[.5,.2,.12,.12,.06]),
 "status": rng.choice(["Delivered","Returned","Cancelled","Pending"], N, p=[.78,.08,.08,.06]),
 "delivery_days": rng.integers(1,12,N).astype(float),
 "rating": rng.choice([1,2,3,4,5], N, p=[.05,.07,.15,.33,.40]).astype(float),
}).astype(object)

def pick(p): return rng.random(N) < p
def setv(col, mask, vals): df.loc[mask, col] = vals if np.isscalar(vals) else vals[:mask.sum()]

# 1. city mess
m = pick(.35); df.loc[m,"city"] = [random.choice(city_vars[c]) for c in df.loc[m,"city"]]
# 2. mixed date formats
fmts = ["%Y-%m-%d","%d/%m/%Y","%m-%d-%Y","%d %b %Y","%B %d, %Y","%d.%m.%y"]
df["order_date"] = [d.strftime(random.choice(fmts)) if random.random()<.45 else d.strftime("%Y-%m-%d") for d in pd.to_datetime(df["order_date"])]
# 3. price as text with Rs. and commas
m = pick(.15); df.loc[m,"unit_price"] = [random.choice(["Rs. {:,}","PKR {:,}","Rs{:,}","{:,}"]).format(p) for p in df.loc[m,"unit_price"]]
# 4. discount as percent strings
m = pick(.2); df.loc[m,"discount"] = [f"{int(d*100)}%" for d in df.loc[m,"discount"]]
# 5. status case/whitespace
m = pick(.25); df.loc[m,"status"] = [random.choice([s.upper(), s.lower(), " "+s, s+" "]) for s in df.loc[m,"status"]]
# 6. quantity words / negatives / zero / outliers
m = pick(.01); df.loc[m,"quantity"] = [random.choice(["one","two","three","two "]) for _ in range(m.sum())]
m = pick(.004); df.loc[m,"quantity"] = -1
m = pick(.002); df.loc[m,"quantity"] = 0
m = pick(.0008); df.loc[m,"quantity"] = 9999
m = pick(.0008); df.loc[m,"unit_price"] = 99999999
# 7. missing values in many encodings
for col,p in [("email",.06),("phone",.05),("rating",.18),("delivery_days",.07),("payment_method",.02),("city",.02),("customer_name",.01)]:
    m = pick(p); df.loc[m,col] = [random.choice([np.nan,"","N/A","null","NULL","-","?","na"]) for _ in range(m.sum())]
# 8. phone formats
m = pick(.3); df.loc[m,"phone"] = [("+92"+str(p)[1:]) if str(p).startswith("03") and random.random()<.5 else (str(p)[:4]+"-"+str(p)[4:] if str(p).startswith("03") else p) for p in df.loc[m,"phone"]]
# 9. invalid emails
m = pick(.02); df.loc[m,"email"] = [random.choice(["abc@","no_email","user@@gmail.com","ali.khan.gmail.com","@yahoo.com"]) for _ in range(m.sum())]
# 10. impossible values
m = pick(.003); df.loc[m,"rating"] = [random.choice([0,6,10,-1,55]) for _ in range(m.sum())]
m = pick(.003); df.loc[m,"delivery_days"] = [random.choice([-3,0,365,999]) for _ in range(m.sum())]
# 11. future / ancient dates
m = pick(.002); df.loc[m,"order_date"] = [random.choice(["2031-05-14","2099-01-01","1970-01-01","0000-00-00","31/02/2024"]) for _ in range(m.sum())]
# 12. customer name inconsistency
m = pick(.08); df.loc[m,"customer_name"] = [str(n).upper() if random.random()<.5 else str(n).lower() for n in df.loc[m,"customer_name"]]
m = pick(.03); df.loc[m,"customer_name"] = ["  "+str(n)+"  " for n in df.loc[m,"customer_name"]]
# 13. category typos
m = pick(.04); df.loc[m,"category"] = [random.choice(["electronics","Electronic","FASHION","home ","Beauty "]) if random.random()<.5 else c for c in df.loc[m,"category"]]
# 14. duplicates (exact + near-dup with new id)
dup = df.sample(frac=.03, random_state=1)
near = df.sample(frac=.01, random_state=2).copy(); near["order_id"] = near["order_id"] + 5_000_000
df = pd.concat([df, dup, near]).sample(frac=1, random_state=3).reset_index(drop=True)

df.to_csv("messy_ecommerce_orders.csv", index=False)
# 15. corrupt lines: extra column, junk footer, junk header comment
with open("messy_ecommerce_orders.csv","r",encoding="utf-8") as f: lines = f.read().split("\n")
for i in random.sample(range(1,len(lines)-1), 40): lines[i] += ",EXTRA,DATA"
lines.append("TOTAL,,,,,,,,,,,,,,,,")
lines.append("Exported by system v2.1 - do not edit")
open("messy_ecommerce_orders.csv","w",encoding="utf-8").write("\n".join(lines))
print(len(df), "rows written")
