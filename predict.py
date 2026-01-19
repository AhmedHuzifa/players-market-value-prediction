import pickle
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
import uvicorn 
from typing import Optional, Literal, Dict, Any
import pandas as pd
import numpy as np
from datetime import date



app = FastAPI(title="player_price_prediction")


class PlayerRequest(BaseModel):
    
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    league: str
    season: int
    team: str
    player: str
    nation: str
    pos: str
    age: int = Field(..., ge=0)
    born: int = Field(..., ge=1800, le=2100)
    mp: int = Field(..., ge=0)
    starts: int = Field(..., ge=0)
    min: int = Field(..., ge=0)
    s90: float = Field(..., ge=0, alias="90s")
    gls: int = Field(..., ge=0)
    ast: int = Field(..., ge=0)
    g_plus_a: int = Field(..., ge=0, alias="g+a")
    g_minus_pk: int = Field(..., ge=0, alias="g-pk")
    pk: int = Field(..., ge=0)
    pkatt: int = Field(..., ge=0)
    crdy: int = Field(..., ge=0)
    crdr: int = Field(..., ge=0)
    xg: float = Field(..., ge=0)
    npxg: float = Field(..., ge=0)
    xag: float = Field(..., ge=0)
    npxg_plus_xag: float = Field(..., ge=0, alias="npxg+xag")
    prgc: int = Field(..., ge=0)
    prgp: int = Field(..., ge=0)
    prgr: int = Field(..., ge=0)
    gls_per90: float = Field(..., ge=0)
    ast_per90: float = Field(..., ge=0)
    g_plus_a_per90: float = Field(..., ge=0, alias="g+a_per90")
    g_minus_pk_per90: float = Field(..., ge=0, alias="g-pk_per90")
    g_plus_a_minus_pk_per90: float = Field(..., ge=0, alias="g+a-pk_per90")
    xg_per90: float = Field(..., ge=0)
    xag_per90: float = Field(..., ge=0)
    xg_plus_xag_per90: float = Field(..., ge=0, alias="xg+xag_per90")
    npxg_per90: float = Field(..., ge=0)
    npxg_plus_xag_per90: float = Field(..., ge=0, alias="npxg+xag_per90")
    other_positions: Optional[str] = Field(None, alias="other positions")
    contract_expiration: Optional[str] = Field(None, alias="contract expiration")
    years_remaining: float = Field(..., ge=0)

    def to_raw_player_dict(self) -> Dict[str, Any]:

        #Returns a dict with the ORIGINAL dataset column names (aliases)
        
        return self.model_dump(by_alias=True)


class PriceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicted_value: float
    
    
  

with open("model.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)



def player_preprocessing(player):
    df_player = pd.DataFrame([player]).copy()
    numerical_final = ["age", "years_remaining", "prgc", "npxg+xag"] 
    categorical_final = ["team", "pos"]
    df_player = df_player[numerical_final + categorical_final]
    player_dict = df_player.to_dict(orient="records")
    return player_dict


def predict_singel(player):
    player_dict = player_preprocessing(player)
    x = dv.transform(player_dict)
    price_log = model.predict(x)[0]
    price_euros = float(np.expm1(price_log))
    return price_euros


@app.post("/predict")
def predict(player: PlayerRequest) -> PriceResponse:
    player_dict = player.to_raw_player_dict()
    price = predict_singel(player_dict)
    
    return PriceResponse(
        predicted_value= float(price)
    )
    
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
