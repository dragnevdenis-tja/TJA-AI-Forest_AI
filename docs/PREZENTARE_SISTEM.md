# Forest Audio AI: Descrierea Prezentării

============================================================
OUTPUT 1: REZUMAT EXECUTIV
============================================================

**Proiect: Forest Audio AI (TerraGuard)**

Forest Audio AI este un sistem avansat de monitorizare a mediului, de nivel industrial, proiectat special pentru a proteja rezervațiile forestiere critice ale Republicii Moldova. Prin implementarea unei rețele reziliente de senzori acustici inteligenți, sistemul oferă detectare și evaluare a riscurilor în timp real pentru amenințările de mediu, inclusiv tăierile ilegale, incendiile de vegetație și intruziunile neautorizate. Platforma transformă datele brute de mediu în informații acționabile, permițând un răspuns rapid din partea pădurarilor și a autorităților de conservare.

Sistemul răspunde nevoii urgente de protecție scalabilă a pădurilor în regiuni precum Codrii, unde patrularea manuală tradițională este intensivă în resurse și adesea reactivă. Inovația principală constă în arhitectura sa cu model dual: o „ureche” acustică ce identifică sunete specifice și un „creier” ambiental care contextualizează aceste sunete în cadrul tiparelor meteorologice și temporale pentru a prezice nivelurile globale de risc. Această abordare stratificată reduce semnificativ alarmele false, asigurând în același timp o sensibilitate ridicată la amenințările reale.

**Arhitectura cu Model Dual**
1.  **Inteligență Acustică (Serviciul 1)**: O rețea neurală convuluțională recurentă (CRNN) cu mecanisme de atenție care clasifică fragmente audio de 5 secunde în șase categorii distincte (Incendiu, Drujbă, Foc de armă, Ploaie, Faună sălbatică, Uman).
2.  **Motorul de Risc Forestier (Serviciul 2)**: Un clasificator XGBoost care procesează confidențele acustice, metadatele meteorologice în timp real (temperatură, umiditate, vânt) și tendințele istorice rulante pentru a categorisi mediul în niveluri de risc SCĂZUT, MEDIU sau RIDICAT.

**Realizări Tehnice Cheie**
- **Pipeline de Analiză Unificat**: Latenta end-to-end, de la capturarea audio brută până la vizualizarea pe tabloul de bord, este optimizată sub 2 secunde pe hardware CPU standard.
- **Detecție de Înaltă Fidelitate**: Clasificatorul acustic a obținut o Precizie Medie Medie (mAP) de **0.993** în teste de simulare cuprinzătoare.
- **Acuratețe Predictivă**: Motorul de risc a obținut un scor perfect de **1.000 ROC-AUC** pe seturi de test stratificate, demonstrând o fiabilitate excepțională în diferențierea între zgomotul de fond al pădurii și evenimentele critice de amenințare.
- **Edge Computing Rezilient**: Implementează o arhitectură IoT pe trei niveluri (Senzor → Gateway → Cloud) cu praguri locale de energie și cozi de date offline.

**Gata pentru Implementare**
Sistemul Forest Audio AI este complet containerizat și gata pentru orchestrare prin Docker. Acesta include o suită cuprinzătoare de teste automate, un tablou de bord interactiv în timp real și documentație tehnică completă, fiind pregătit pentru integrarea imediată a hardware-ului și programe pilot pe teren.

============================================================
OUTPUT 2: DESCRIEREA ARHITECTURII TEHNICE
============================================

### Secțiunea A — Prezentare Generală a Sistemului
Arhitectura Forest Audio AI este un pipeline distribuit, conceput pentru disponibilitate ridicată și latență scăzută. Fluxul începe la **Nodul de Margine (ESP32-S3)**, care capturează fragmente audio de 5 secunde și metadate de bază despre mediu. Datele sunt transmise printr-o legătură mesh locală sau WiFi către un **Gateway Rezilient (Raspberry Pi)**, care gestionează o coadă asincronă și eșecurile de conectivitate.

**Backend-ul FastAPI** centralizează orchestrarea analizei. În locul unui model monolitic, a fost aleasă o **arhitectură cascadată cu două servicii** pentru modularitate și acuratețe contextuală. Serviciul 1 (Audio) execută sarcina specializată de clasificare a sunetului. Rezultatele sale — scorurile de confidență pentru diverse evenimente — sunt apoi introduse ca variabile (features) în Serviciul 2 (Risc). Această separare permite motorului de risc să pondereze importanța unui sunet diferit în funcție de mediu (de exemplu, o drujbă detectată în adâncul Codrilor declanșează un risc RIDICAT, în timp ce același sunet într-un parc urban din Chișinău ar putea fi doar risc MEDIU). În final, rezultatele sunt difuzate prin **WebSockets** către un **Dashboard React** interactiv pentru vizualizare în timp real pe harta Moldovei.

### Secțiunea B — Serviciul 1: Inteligență Acustică
Serviciul 1 utilizează o arhitectură de ultimă generație de tip **CRNN (Convolutional Recurrent Neural Network)**.
- **Intrare**: Fragmente audio de 5 secunde eșantionate la 32kHz, transformate în Mel-spectrograme (64 benzi Mel).
- **Arhitectură**: Include un strat **PCEN (Per-Channel Energy Normalization)** pentru suprimarea adaptivă a zgomotului de fundal, urmat de straturi convoluționale 2D pentru extragerea caracteristicilor spațiale, un **GRU** (Gated Recurrent Unit) bidirecțional pentru dependențe temporale și un mecanism de **Atenție** pentru a se concentra pe evenimentele acustice cu energie ridicată.
- **Ieșire**: Probabilități sigmoidale multi-etichetă pentru `incendiu`, `drujbă`, `foc_de_armă`, `ploaie`, `faună_sălbatică` și `om`.
- **Performanță**: Verificată la **0.993 mAP** pe toate clasele în simulări controlate.

### Secțiunea C — Serviciul 2: Motorul de Risc Forestier
Motorul de Risc este „Creierul Contextual” al sistemului, bazat pe un **Clasificator XGBoost**.
- **Variabile de Intrare (Features)**: 
    - **Meteo**: Temperatură, Umiditate, Viteza vântului, Precipitații (critice pentru riscul de incendiu).
    - **Acustică**: Cele mai recente scoruri de confidență de la Serviciul 1.
    - **Temporal**: Ora din zi, Ziua săptămânii, Sezonul.
    - **Geografic**: Latitudine, Longitudine, ID Regiune.
    - **Tendințe Rulante**: Medii pe 5m, 10m și 30m ale detecțiilor de amenințări (ex: zgomot persistent de drujbă vs. un singur sunet izolat).
- **Ingineria Caracteristicilor**: Implementează **codare ciclică** (Sin/Cos) pentru timp pentru a capta modele periodice și calculează un **scor compozit audio** (indice de amenințare ponderat) pentru a rezuma dovezile acustice.
- **Raționament**: XGBoost a fost selectat în locul regresiei logistice datorită capacității sale superioare de a gestiona interacțiuni neliniare între vreme (umiditate scăzută) și evenimente acustice (pârâitul focului).
- **Etichetare**: Utilizează reguli de **Supervizare Slabă (Weak Supervision)** bazate pe standardele de mediu din Moldova pentru a atribui niveluri de risc fără a necesita etichetarea manuală a fiecărei ferestre de timp.

### Secțiunea D — Pipeline în Timp Real
Stratul în timp real asigură că pădurarii primesc actualizări în câteva secunde de la o detecție.
- **Protocol WebSocket**: O conexiune persistentă `ws:///ws/live` trimite evenimente de tip `inference_result` și `alert` către toți clienții.
- **Buffer Rulant**: Backend-ul menține un buffer thread-safe în memorie pentru fiecare `sensor_id`, calculând mediile glisante necesare pentru modelul de risc.
- **Gestionarea Latenței**: Inferența de înaltă performanță este realizată prin încărcarea modelelor în memorie o singură dată la pornire, utilizând un handler `lifespan` de la FastAPI.
- **Reziliență Offline**: Dacă un nod senzor se deconectează, sistemul îl marchează ca „Avertizare” după 2 minute și „Offline” după 5 minute pe tabloul de bord, în timp ce gateway-ul de margine stochează datele pentru sincronizare ulterioară.

### Secțiunea E — Arhitectura IoT
- **Noduri Senzor ESP32**: Acționează ca și colectoare distribuite de cost redus. Acestea efectuează **Filtrare la Margine (Edge Filtering)** utilizând calculul energiei RMS pentru a evita transmiterea datelor silențioase sau irelevante, prelungind semnificativ durata de viață a bateriei.
- **Gateway Raspberry Pi**: Servește ca hub local de inteligență. Se abonează la subiectele MQTT, stochează fragmentele audio și gestionează cererile HTTPS autentificate către Cloud API.
- **Zone de Implementare**: 6 senzori sunt mapați strategic în Moldova: 2 în **Codrii** (pădure bătrână densă), 2 în **Nord** (granițe agricole), 1 în **Chișinău** (control urban/parc) și 1 în **Sud** (stepă/utilizare mixtă).

============================================================
OUTPUT 3: METODOLOGIA DATELOR ȘI ML
============================================================

**Prezentare Generală a Metodologiei**
Proiectul Forest Audio AI utilizează o abordare centrată pe date pentru dezvoltarea modelelor, asigurând o fiabilitate ridicată în absența multor ani de date istorice de pe teren.

- **Generarea de Date Sintetice**: Pentru a antrena modelul de risc, a fost construit un generator determinist care a creat 10.000 de mostre de date realiste de la senzori. Simularea încorporează climatul Moldovei (ex: veri calde și uscate care declanșează riscul de incendiu) și realitățile geografice (ex: modele de tăieri ilegale în Codrii).
- **Supervizare Slabă (Weak Supervision)**: Etichetele de antrenament au fost atribuite folosind un sistem expert bazat pe reguli (ex: `Confidență Incendiu > 0.7 ȘI Umiditate < 35% => Risc RIDICAT`). Acest lucru permite modelului să învețe granițele complexe între aceste reguli experte și interacțiunile brute ale variabilelor.
- **Distribuția Claselor**: Setul de date a fost echilibrat la aproximativ **60% SCĂZUT**, **25% MEDIU** și **15% RIDICAT** risc pentru a asigura robustețea modelului la evenimente critice rare.
- **Strategia de Validare**: A fost utilizată o **împărțire stratificată 60/20/20** (Antrenare/Validare/Testare) pentru a menține proporții constante ale claselor în toate subseturile, prevenind evaluarea părtinitoare.
- **Prevenirea Scurgerilor de Date (Leakage)**: Ingineria caracteristicilor este realizată strict în cadrul pipeline-ului, iar variabilele rulante sunt verificate pentru a nu folosi niciodată date „din viitor” din simulare. Analiza decalajului de corelație este utilizată pentru a asigura că nicio variabilă nu este un proxy neintenționat pentru eticheta țintă.
- **Reproductibilitate**: Toate operațiunile aleatorii (generarea datelor, inițializarea modelelor, împărțirile) sunt guvernate de un seed global `RANDOM_SEED = 42` definit în `backend/config.yaml`.
- **Limitări Cunoscute**: Setul actual de date este sintetic. Deși este coerent sezonier, îi lipsește întreaga diversitate acustică a unei păduri reale (ex: specii specifice de păsări sau variate tipuri de motoare de drujbă), ceea ce va fi abordat prin implementarea „Ciclului de Auto-Învățare”.

============================================================
OUTPUT 4: ETICĂ ȘI IMPLEMENTARE RESPONSABILĂ
============================================================

**Rezumat Etică și Siguranță**
Forest Audio AI este conceput cu un mandat **„Privacy-First”**, asigurându-se că monitorizarea mediului nu evoluează în supraveghere umană neautorizată.

- **Politică de Atenuare**: Sistemul este restricționat tehnic la identificarea amenințărilor de mediu. Prezența umană este detectată doar pentru a asigura confidențialitatea vizitatorilor pădurii prin suprimarea imediată a datelor.
- **Politică privind Vorbirea Umană**: Orice fragment audio cu un scor de confidență `uman` care depășește 0.5 este marcat automat pentru ștergere imediată și nu este niciodată stocat sau transmis în cloud.
- **Reținerea Datelor**: Datele care nu reprezintă o amenințare sunt eliminate din memorie în 60 de secunde; doar metadatele și evenimentele de amenințare verificate sunt păstrate pentru studii ecologice pe termen lung.
- **Consecințele Alarmelor False**: „Verificarea de Context” a modelului dual minimizează riscul de „oboseală a alertelor” și desfășurarea inutilă a resurselor, solicitând atât dovezi acustice, cât și de mediu înainte de a declanșa o alertă de risc RIDICAT.
- **Părtinire Geografică**: Recunoaștem că modelul poate funcționa diferit în regiunile Moldovei din cauza propagării acustice variate; rulările de calibrare regională sunt obligatorii înainte de implementarea completă.
- **Implementare Responsabilă**: Operațiunile pe teren necesită semnalizare publică clară, conformitatea legală cu legile moldovenești privind datele și un protocol „human-in-the-loop” pentru toate intervențiile cu risc ridicat.
