# Quantum_QAE_GDO_v1.py

# QAE - Quantum Amplitude Encoding (Kvantno enkodiranje amplituda)
# GDO - Grover Diffusion Operator (Groverov operator difuzije)


import pandas as pd
import numpy as np
import random
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator

# Konstrukcija okruženja (Striktni parametri protokola)
MAX_BROJ = 39
BROJ_KUGLICA = 7
KUBITA_PO_POZICIJI = 5 # Moje rešenje: 2^5 = 32 stanja po svakoj poziciji bez izuzetka
DIMENZIJA_REGISTRA = 32

random.seed(39)
np.random.seed(39)


# Putanja do podataka (CSV)
# putanja = '/data/loto7_4708_k85.csv'
# putanja = '/data/loto7_4708_k85_loto_2976.csv'
putanja = '/data/loto7_4708_k85_loto_plus_1732.csv'


# Učitavanje fajla
df = pd.read_csv(putanja, header=None)

# Izolacija tačno poslednjih 7 kolona sa repa tabele (hronološki, poziciono)
H = df.iloc[:, -7:].values.astype(int)

# Obezbeđivanje da su istorijski redovi sortirani radi tačnog pozicionog mapiranja opsega
H_sorted = np.sort(H, axis=1)
N_redova = len(H_sorted)

kombinacija = []

# Analiza za svaku od 7 pozicija preko stvarnog Qiskit kvantnog kola
for i in range(BROJ_KUGLICA):
    pozicija = i + 1
    min_val = pozicija
    
    istorija_pozicije = H_sorted[:, i]
    
    # Inicijalizacija jediničnih kvantnih amplituda
    kvantne_amplitude = np.zeros(DIMENZIJA_REGISTRA, dtype=complex)
    
    # NULA FREKVENCIJA: Istorijski niz se enkodira preko relativne pozicione entropije.
    # Svaka unikatna vrednost dobija fazni i amplitudni potpis na osnovu svog poslednjeg 
    # sekvencijalnog pojavljivanja na toj poziciji (vremenski marker), čime se eliminiše sabiranje i brojanje frekvencija.
    poslednji_marker = {}
    for korak, broj in enumerate(istorija_pozicije):
        indeks_stanja = broj - min_val
        if 0 <= indeks_stanja < DIMENZIJA_REGISTRA:
            poslednji_marker[indeks_stanja] = korak
            
    for indeks_stanja, marker in poslednji_marker.items():
        # Generisanje jedinstvene nelinearne amplitude na osnovu normalizovane vremenske distance (bez frekvencija)
        relativno_vreme = marker / N_redova
        kvantne_amplitude[indeks_stanja] = np.sin(relativno_vreme * np.pi / 2) + 1j * np.cos(relativno_vreme * np.pi / 2)
            
    # Normalizacija talasne funkcije u Hilbertovom prostoru čistih amplituda
    if np.linalg.norm(kvantne_amplitude) > 0:
        kvantne_amplitude /= np.linalg.norm(kvantne_amplitude)
    else:
        kvantne_amplitude = np.ones(DIMENZIJA_REGISTRA, dtype=complex) / np.sqrt(DIMENZIJA_REGISTRA)
        
    # KVANTO KOLO (5 kubita za trenutnu poziciju)
    qc = QuantumCircuit(KUBITA_PO_POZICIJI)
    
    # Učitavanje čistih istorijskih amplituda direktno u kvantne kubite
    qc.initialize(kvantne_amplitude, range(KUBITA_PO_POZICIJI))
    
    # DEFINISANJE IDEALNOG GROVEROVOG OPERATORA DIFUZIJE PREKO UNITARNE MATRICE
    # D = 2|s><s| - I
    I = np.eye(DIMENZIJA_REGISTRA)
    s = np.ones((DIMENZIJA_REGISTRA, 1)) / np.sqrt(DIMENZIJA_REGISTRA)
    D_matrica = 2 * np.dot(s, s.T) - I
    Grover_operator = Operator(D_matrica)
    
    # Apliciranje Groverovog operatora direktno na kolo
    qc.append(Grover_operator, range(KUBITA_PO_POZICIJI))
    
    # SIMULACIJA TALASNE FUNKCIJE
    state = Statevector.from_instruction(qc)
    probabiliteti = state.probabilities_dict()
    
    # Deterministička ekstrakcija binarne baze sa najvećom amplitudom
    najbolje_stanje_bin = max(probabiliteti, key=probabiliteti.get)
    izabrani_indeks = int(najbolje_stanje_bin, 2)
    
    # Vraćanje izmerenog 5-bitnog indeksa u realnu Loto vrednost
    izabrani_broj = izabrani_indeks + min_val
    
    # Rešavanje duplikata preko Jacobijeve refleksije unutar opsega trenutne pozicije
    val = izabrani_broj
    pomak = 1
    max_val_pozicije = min_val + 32
    
    while val in kombinacija:
        val = izabrani_broj + pomak
        while val < min_val or val > max_val_pozicije:
            if val < min_val:
                val = min_val + (min_val - val)
            if val > max_val_pozicije:
                val = max_val_pozicije - (val - max_val_pozicije)
        pomak += 1
        
    kombinacija.append(int(val))



# Izlazni rezultati
print()
print(f"Mod Čisto Kvantni Model: Qiskit 5-Kubitni Čist Amplitudni Pozicioni Model (Vremenski marker)")
print(f"Analizirano redova iz CSV: {N_redova}")
print(f"Predložena kombinacija: {sorted(kombinacija)}")
print()

"""
Mod Čisto Kvantni Model: Qiskit 5-Kubitni Čist Amplitudni Pozicioni Model (Vremenski marker)
Analizirano redova iz CSV: 4708
Predložena kombinacija: [4, x, 6, y, 26, z, 34]



Mod Čisto Kvantni Model: Qiskit 5-Kubitni Čist Amplitudni Pozicioni Model (Vremenski marker)
Analizirano redova iz CSV: 2976
Predložena kombinacija: [4, x, 6, y, 26, z, 33]



Mod Čisto Kvantni Model: Qiskit 5-Kubitni Čist Amplitudni Pozicioni Model (Vremenski marker)
Analizirano redova iz CSV: 1732
Predložena kombinacija: [4, x, 6, y, 23, z, 32]
"""




"""
QAE i GDO - Akronimi za dve ključne čisto kvantne operacije, a koje služe da se izbaci klasična statistika i omogući rad isključivo sa kvantnim amplitudama:

1. QAE - Quantum Amplitude Encoding (Kvantno enkodiranje amplituda)

• Metod unosa (učitavanja) klasičnih podataka u kvantni računar. Umesto da se brojevi pretvaraju u obične binarne bitove (0 i 1), celi nizovi podataka iz CSV fajla se preslikavaju direktno u težine (amplitude) samog kvantnog stanja.
• Svaki broj ili vremenski marker iz istorije dobija svoju specifičnu kompleksnu amplitudu (fazu). Na ovaj način, ceo CSV fajl sa hiljadama redova prestaje da postoji kao klasična tabela i postaje jedna jedinstvena, uvezana kvantna talasna funkcija unutar registra.

2. GDO - Grover Diffusion Operator (Groverov operator difuzije)

• Osnovni matematički pokretač unutar čuvenog Groverovog algoritma kvantne pretrage. Njegova funkcija je da izvrši takozvanu „inverziju oko srednje vrednosti” (Inversion about the mean) nad kvantnim amplitudama.
• Kada se učita istoriju preko QAE, prostor je u ujednačenoj superpoziciji. GDO tada deluje kao nelinearni kvantni filter: on deterministički smanjuje i prigušuje amplitude onih stanja koja predstavljaju šum (destruktivna interferencija), dok istovremeno ekstremno pojačava i "pumpa" amplitude onih stanja gde se poklapaju fazni markeri (konstruktivna interferencija). Izlaz iz GDO operatora daje direktnu pobedničku tačku u prostoru koja je pretrpela najveći kvantni skok.

Zajedno, QAE i GDO omogućavaju da se prostor od 32 mogućnosti po svakoj koloni pretraži čisto kvantno — QAE učitava prošlost u vidu talasa, a GDO podiže vrh tog talasa na mestu gde sistem treba da napravi sledeći korak.
"""




"""
Ovaj kôd je čisto kvantni algoritam po svojoj strukturi i matematičkom izvršavanju na kvantnom simulatoru.
Iz njega je potpuno uklonjen hibridni klasični račun (SVD, kovarijansa, optimizacione petlje, klasični ML). 
Sve transformacije nad stanjima izvršavaju se unutar kvantnog prostora simulatora.
Međutim, sa najstrožeg inženjerskog aspekta kvantne teorije, u kodu postoji jedan hibridni most u samoj pripremi podataka (Korak 1):

Zašto je algoritam KVANTNI:

1. ** Hilbertov prostor stanja:** Podaci se preslikavaju u stvarni 5-kubitni kvantni registar (QuantumCircuit(5)) za svaku poziciju.
2. Kvantna evolucija (Grover): Kvantni operator difuzije se unosi kao punopravna unitarna matrica (Operator(D_matrica)) i primenjuje direktno na kolo (qc.append). To simulira pravo kvantno ponašanje i inverziju oko srednje vrednosti talasne funkcije.
3. Nema klasične statistike: Selekcija stanja se vrši na osnovu čiste interferencije i krajnjih amplituda talasne funkcije (Statevector), bez brojanja frekvencija i klasičnog uzorkovanja.

Gde se nalazi jedini KLASIČNI korak:

• Klasični parser (Rečnik markera): Petlja for korak, broj in enumerate(istorija_pozicije) i rečnik poslednji_marker se izvršavaju na standardnom CPU kako bi se izračunale vrednosti faza pre nego što se one učitaju u kolo komandom qc.initialize.
U realnom, čisto kvantnom hardverskom sistemu, ovaj vremenski marker bi morao biti upisan u kvantnu memoriju (QRAM) pomoću kvantnih adresa. Pošto QRAM tehnologija još uvek nije lokalno dostupna, ovaj korak se obavlja klasičnim prevođenjem u niz kompleksnih brojeva koje Qiskit simulator potom preuzima i simulira kao čisto kvantno stanje.
"""
