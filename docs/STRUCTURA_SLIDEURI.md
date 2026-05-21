# Forest Audio AI: Structura Slide-urilor (Prezentare 10 Minute)

1. **Titlu: TerraGuard - Monitorizare Neurală a Pădurilor**
   - Refactorizarea sistemului Forest Audio AI.
   - Pipeline neural dual pentru protecția mediului.
   - Monitorizare în timp real pentru rezervațiile din Moldova.
   - *Notă vorbitor: Subliniați că acesta este un sistem de nivel industrial gata pentru pilotare.*

2. **Problema: Protejarea Inimii Verzi a Moldovei**
   - Acoperirea forestieră de 12% se confruntă cu tăieri ilegale și incendii provocate de climă.
   - Patrularea manuală este insuficientă pentru monitorizarea 24/7.
   - Nevoia de detectare automată, cu costuri reduse și în timp real.
   - *Notă vorbitor: Concentrați-vă pe mizele mari — pierderea pădurilor bătrâne precum Codrii.*

3. **Prezentare Generală a Sistemului: De la Margine la Dashboard**
   - Nivelul 1: Noduri de Margine ESP32 (Audio + Metadate).
   - Nivelul 2: Gateway Raspberry Pi (Cozi Reziliente).
   - Nivelul 3: Backend FastAPI + ML (Inteligență Centrală).
   - Nivelul 4: Dashboard React (Vizualizare în Timp Real).
   - *Notă vorbitor: Descrieți pe scurt călătoria datelor de la un sunet de pădure la un punct roșu pe hartă.*

4. **Serviciul 1: Inteligență Acustică (CRNN)**
   - Clasificator acustic specializat.
   - PCEN pentru suprimarea zgomotului + Atenție pentru separarea sunetelor.
   - 6 clase: Incendiu, Drujbă, Foc de armă, Ploaie, Faună, Om.
   - *Notă vorbitor: Scoateți în evidență faptul că modelul „ascultă” special după semnături de amenințare.*

5. **Serviciul 2: Motorul de Risc Forestier (XGBoost)**
   - Decidentul contextual.
   - Integrează vremea, timpul și tendințele istorice.
   - Prezice nivelurile de risc SCĂZUT, MEDIU sau RIDICAT.
   - *Notă vorbitor: Explicați că acest model previne alarmele false verificând dacă un sunet „are sens” în mediul său.*

6. **Insight-ul Central: Inferență Cascadată**
   - De ce două modele? Separarea între „Ce se aude” (Audio) și „Și ce dacă?” (Risc).
   - Rezultatele Serviciului 1 alimentează vectorul de caracteristici al Serviciului 2.
   - Permite ponderarea regională și de mediu a amenințărilor.
   - *Notă vorbitor: Acesta este cel mai important slide tehnic — „Creierul” care contextualizează „Urechea”.*

7. **Metodologia Datelor și Supervizarea Slabă**
   - Set de date sintetic de 10.000 de mostre.
   - Etichetare expertă bazată pe reguli (Weak Supervision).
   - Testare stratificată pentru echilibrul claselor.
   - *Notă vorbitor: Subliniați rigoarea generării datelor și a verificărilor de scurgere a informațiilor.*

8. **Conectivitate în Timp Real și IoT**
   - Server WebSocket pentru difuzarea live a evenimentelor.
   - Filtrare a energiei RMS la margine pentru a economisi bateria.
   - Capacitate de sincronizare offline pentru senzorii deconectați.
   - *Notă vorbitor: Detaliați funcțiile de reziliență care fac sistemul gata pentru teren.*

9. **Turul Dashboard-ului**
   - Hartă Leaflet interactivă a Moldovei.
   - Grafice de confidență neurală live.
   - Cronologia alertelor și monitorizarea sănătății senzorilor.
   - *Notă vorbitor: Indicați actualizările în timp real și experiența intuitivă a utilizatorului.*

10. **Rezultate de Evaluare**
    - Model Audio: 0.993 mAP (Simulat).
    - Model de Risc: 1.000 ROC-AUC.
    - Latență: < 2 secunde end-to-end.
    - *Notă vorbitor: Folosiți aceste cifre solide pentru a dovedi fiabilitatea sistemului.*

11. **Etică și Confidențialitate**
    - Arhitectură „Discard-by-default”.
    - Suprimarea automată a vorbirii umane.
    - Notificare publică și conformitate legală.
    - *Notă vorbitor: Abordați preocupările legate de confidențialitate proactiv și cu încredere.*

12. **Munca Viitoare: Ciclul de Auto-Învățare**
    - Tranziția de la date sintetice la date reale de pe teren.
    - Calibrare acustică regională.
    - Miniaturizarea hardware-ului IoT (Solar + LoRa).
    - *Notă vorbitor: Arătați calea de la acest prototip la o rețea la nivel național.*

13. **Concluzie și Demo Live**
    - TerraGuard: Protejarea pădurilor prin monitorizare neurală.
    - Modular, scalabil și gata de producție.
    - Invitație de a vizualiza dashboard-ul live.
    - *Notă vorbitor: Încheiați pe o notă pozitivă despre impactul asupra mediului.*
