import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
LABELS={"label","target","class","anomaly","is_anomaly"}
def demo_dataframe(seed=7,n=80):
 r=np.random.default_rng(seed); X=np.vstack([r.normal(0,1,(n,4)),r.normal(4.5,.7,(8,4))])
 d=pd.DataFrame(X,columns=["event_count","sequence_length","duration","error_rate"]); d["label"]=["Normal"]*n+["Anomaly"]*8; return d
def predict_dataframe(df):
 label=next((c for c in df.columns if c.strip().lower() in LABELS),None)
 X=df.select_dtypes(include=[np.number]).copy()
 if label in X.columns:X=X.drop(columns=[label])
 X=X.replace([np.inf,-np.inf],np.nan).dropna(axis=1,how="all")
 if X.empty: raise ValueError("No numeric feature columns found; convert logs into numeric features first.")
 model=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),IsolationForest(n_estimators=150,contamination="auto",random_state=42)); model.fit(X)
 p=model.predict(X); out=df.copy(); out["prediction"]=np.where(p==-1,"Anomaly","Normal"); out["anomaly_score"]=(-model.decision_function(X)).round(6)
 summary={"rows":len(out),"anomalies":int((out.prediction=="Anomaly").sum()),"method":"Isolation Forest baseline (unsupervised; numeric features only)"}
 return out,summary
