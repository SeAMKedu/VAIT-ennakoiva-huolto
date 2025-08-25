![Älykkäät teknologiat](https://storage.googleapis.com/seamk-production/2022/04/2b1d63e0-alykkaat-teknologiat_highres_2022-768x336.jpg)
![EPLiitto logo](https://github.com/SeAMKedu/VAIT-tekoaly-rag/blob/main/kuvat/EPLiitto_logo_vaaka_vari.jpg)
![EU osarahoitus](https://github.com/SeAMKedu/VAIT-tekoaly-rag/blob/main/kuvat/FI_Co-fundedbytheEU_RGB_POS.png)

# Tärinädatasta teränvaihdon ennustaminen (RandomForest)

Paikallisesti ajettava Python-pilotti, joka ryhmittelee tärinämittausten aikajonoa minuutti-ikkunoihin, laskee tilasto- ja aikaominaisuuksia (lagit, liukuvat ikkunat, muutosnopeus), opettaa **RandomForestClassifier**-mallin ja ennustaa **teränvaihdon todennäköisyyttä** tulevassa datassa. Koodi piirtää sekä akselikohtaiset tärinän keskihajonnat että vaihdon todennäköisyysajan­jatkumon ja vertaa ennusteita tunnettuihin vaihtohetkiin. Tämä pilotti tehtiin osana vAI:lla tuottavuutta? -hanketta. Tämä pilotti ei tue livedataa, vaan se jätettiin jatkokehitys-listalle. Pilotin tarkoituksena oli pilotoida sitä, onko mahdollista ennustaa teränvaihtoa laitteen kyljestä kerätyllä tärinädatalla ja tuotannosta kerätyn teränvaihdon ajoitusdatasta. Koodi on luotu nimenomaan pilottia ajatellen, eli se ei käytä välttämättä parhaita käytäntöjä, vaan tarkoituksena on näyttää tekoälyn mahdollisia käyttötapoja ja minkälaisia tuloksia voi nykyteknologialla saada aikaan. 

Tärinädata kerättiin koneen ulkokuoreen asetetulla tärinäanturilla. 

Sovelluksia joita käytetään Windows-/Linux-tietokoneella:

- Python 3.9 – 3.13  
- Visual Studio Code  
- (valinnainen) Conda / uv virtuaaliympäristöä varten  
- Git

> Malli on suunniteltu toimimaan ilman pilvipalveluja ja sopii siten yritys­ympäristöihin, joissa dataa ei haluta siirtää ulkoisiin palveluihin.

# Julkaisun historiatiedot

**Merkittävät muutokset julkaisuun**

| pvm        | Muutokset                        | Tekijä           |
|------------|----------------------------------|------------------|
| 13.08.2025 | Versio 1.0 – ensimmäinen julkaisu | Teemu Virtanen   |
| xx.xx.2025 | Zenodo-julkaisu                  | Teemu Virtanen   |

# Sisällysluettelo

- Julkaisun nimi  
- Julkaisun historiatiedot  
- Sisällysluettelo  
- Teknologiapilotti  
- Hanketiedot  
- Kuvaus  
- Tavoitteet  
- Toimenpiteet  
- Asennus ja käyttö  
- Python-ohjelman käyttö  
- Havaitut virheet ja ongelmatilanteet  
- Vaatimukset  
- Tulokset  
- Lisenssi  
- Tekijät

# Tuottavuuspilotti

vAI:lla tuottavuutta? -hankkeen työpaketti 2 tuottaa tuottavuuspilotteja. Tämä pilotti keskittyy **teränvaihdon ennakointiin tärinädatan avulla**.

# Hanketiedot

- **Hankkeen nimi:** vAI:lla tuottavuutta?  
- **Rahoittaja:** Euroopan unionin osarahoittama. Euroopan aluekehitysrahasto (EAKR). Etelä-Pohjanmaan liitto.  
- **Toteuttajat:** Päätoteuttajana Seinäjoen Ammattikorkeakoulu Oy, osatoteuttajina Tampereen korkeakoulusäätiö sr ja Vaasan yliopisto  
- **Aikataulu:** 1.8.2024 – 31.12.2026

# Kuvaus

Skripti lukee `vibration_data_kaikki.csv`-tiedoston (sarake **time** + akselit **x, y, z**), resamplaa datan 1 minuutin ikkunoihin ja laskee ominaisuuksia:

- perus­tilastot: mean, std, min, max, count  
- viiveominaisuudet: 1–3 ikkunan lagit (mean/std)  
- liukuvat ominaisuudet (2 ja 3 ikkunan): std, mean, “range-ratio”  
- muutosnopeudet: std:n suhteellinen muutos (diff / edellinen)  
- aikaominaisuudet: tunnin ja viikonpäivän numerot

Tavoitemuuttuja **blade_change** syntyy merkitsemällä 0–2 h ennen tunnettuja vaihtoja arvo 1. Malli opetetaan aikajärjestyksen mukaan (70 % koulutus, 30 % “tulevaisuus” testiin). Ennusteista poimitaan tapahtumat kynnys- ja cooldown-logiikalla, ja niitä verrataan todellisiin vaihtoihin (TP/FP/missed, precision/recall/F1).

# Tavoitteet

- Näyttää, miten **edullinen, paikallinen** ML-ratkaisu voi antaa ennakkovaroituksia teränvaihdoista.  
- Tarjota **muokattava referenssi**, jossa kynnysten, ikkunoiden ja ominaisuuksien säätö vaikuttaa suoraan havaittuun suorituskykyyn.  
- Tuottaa **visuaaliset aikasarjakäyrät** kunnossapidon analysoitavaksi.

# Toimenpiteet

- Datan esikäsittely ja resamplaus minuuttitasolle.  
- Ominaisuuksien rakentaminen (lagit, rolling-ikkunat, ROC).  
- **RandomForestClassifier** (class_weight='balanced') opetus.  
- Ennustetapahtumien poiminta (`threshold`, `cooldown_period`).  
- Arviointi todellisia vaihtohetkiä vasten (±1 h ikkuna).  
- Visualisoinnit: tärinän std-käyrät, vaihdon todennäköisyys, todelliset vs. ennustetut vertikaaliviivat sekä zoomattu esimerkki oikein osuneesta tapahtumasta.

# Asennus ja käyttö

## Vaatimukset

- Tietokone, jossa suositus **16 GB RAM** (tai vastaava)  
- Vapaata levytilaa datalle ja kuvien tallennukselle

## Asennus (vaihtoehto A: uv)

(valinnainen) asenna uv: https://github.com/astral-sh/uv
uv venv
uv pip install -r requirements.txt

## Asennus (vaihtoehto B: pip/conda)

python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

## Python-ohjelman käyttö
Sijoita vibration_data_kaikki.csv juurikansioon. Vähintään sarake time (ISO-aikaleima) sekä akselit x, y, z.

Avaa VS Code / terminaali, aktivoi venv.

Aja skripti:

bash
Kopioi
Muokkaa
python VAIT-ennakoiva-huolto.py

Konsoli tulostaa:
- ikkunoiden määrät ja aikavälin
- tärkeimmät piirteet (feature_importances)
- TP/FP/missed, precision/recall/F1 (testijakso)

Näytölle piirtyvät kuvat:
- koko testijakson std + probability
- zoomattu ikkuna ensimmäisestä oikein osuneesta tapahtumasta

Parametrien säätö
- window_size = '1T' – resamplaus (esim. '30S', '2T')
- prediction_window=timedelta(hours=2) – kuinka pitkältä ajalta ennen vaihtoa merkitään 1
- threshold (esim. 0.07–0.15) – todennäköisyysraja tapahtumalle
- cooldown_period (esim. 6) – minimietäisyys peräkkäisten tapahtumien välillä (minuutteina, koska ikkuna on 1T)

## Havaitut virheet ja ongelmatilanteet
- CSV puuttuu → tulostuu virheilmoitus ja ohjelma poistuu.
- Aikaleimat: varmista, että time parsitaan oikein ja aikavyöhykkeisyys on yhdenmukainen.
- NaN/inf ominaisuuksissa → koodi pudottaa puuttuvat/äärettömät ennen mallia.

## Vaatimukset
- Python 3.9–3.13
- Kirjastot: pandas, numpy, scikit-learn, matplotlib
- Datan sarakevaatimukset: time, x, y, z
- Tunnetut vaihtolistat (part_changes) annetaan koodissa aikaleimoina (UTC, tai lokalisoidaan tz_localize('UTC')).

## Tulokset
- Tulostuu tarkkuus-, recall- ja F1-arviot testijaksolta.
- Kuvissa näkyy todellisten ja ennustettujen vaihtohetkien kohdistuminen.
- Oletusasetuksilla demo antaa kunnossapidolle alkutason hälytyskelpoisen signaalin; parannettavissa datan laadulla, ominaisuuksien laajennuksilla (esim. spektriominaisuudet), malli- ja raja-arvo-säädöillä.

# Lisenssi
Dokumentaatio ja koodi julkaistaan MIT-lisenssillä (sama lisenssilinja kuin hankkeen muissa demoissa).

## Tekijät
Teemu Virtanen (koodi ja dokumentaatio)

vAI:lla tuottavuutta? -hanketiimi / SEAMK
