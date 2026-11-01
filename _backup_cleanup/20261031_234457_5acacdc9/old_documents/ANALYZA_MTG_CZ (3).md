**APEX-T: rozbor dosavadního skóre a doporučený další krok — 31. 10. 2026**

Navázal jsem na chat „Stanovení skóre podivnosti“ a přečetl implementace
z `APEX-T-M2-fast-labeler.zip`, `APEX-T-MT-fast-labeler.zip` a
`APEX-T-G-geometry-labeler.zip`. Poslední uživatelův závěr je, že T na jeho
měřeních pomáhá nejvíc; přínos M je nejasný a G se příliš neosvědčilo.
To je užitečný výsledek průzkumu, zatím však chybějí dostupné tabulky jeho
ručních štítků pro výpočet metrik. Tento dokument neudává přesnost na reálných datech.

| Metoda | Implementované chování | Co z toho plyne |
|---|---|---|
| M2.0 | Hledá až čtyři přímkové hypotézy ze souřadnic, hodnotí dvojice křížení a rovnoběžných stop, pokrytí a vyváženost podpory. | Je to specifická hypotéza o přímých stopách, nikoliv obecný detektor všech překryvů. Zkřivené nebo krátké stopy mohou zůstat bez vysokého M. Počet fitovaných přímek není počet částic. |
| T1.1 | Ztenčí binární masku původních aktivních pixelů, odstraní krátké koncové větvičky a počítá konce dosahující rozvětvení. | Na vizuálně větvených stopách je přirozeně užitečné. Neověřuje původ větvení ani skutečné množství částic. |
| G1.0 | 80 % změna šířky podél prunovaného skeletonu, 20 % nekonvexnost obrysu. | Přirozeně zachytí i ohyb jediné stopy; nemusí reagovat na tenké překřížení. V těchto měřeních nemusí přidat hodnotu k T/M. |
| A | Původní hypotéza velikosti nebo odlehlosti, nikoliv ověřená finální metrika. | Prozatím ukládat N pixelů a relativní velikost v konkrétním měření. Bez referenčních jednotlivých stop nelze velikost převést na spolehlivý důkaz pile-upu. |

T1.1 zatím dává převážně několik úrovní: tři podporovaná ramena 45,
čtyři 80, pět a více 95 a malý bonus za další skupiny uzlů. Rozdíl mezi
skóre například 45 a 80 je návrhové pravidlo; neznamená změřený rozdíl
pravděpodobností. Současný algoritmus agreguje ramena přes celý klastr,
nikoliv přes explicitně rekonstruované fyzikální trajektorie. Pevné
ořezání větví v počtu kroků navíc může reagovat na velikost a orientaci stopy.

V M je důležitá jeho geometrická podmínka: pro křížení potřebuje podporu
na obou stranách průsečíku na obou přímkách. Y proto obvykle nedostane
vysoké skóre M. To pomáhá odlišit jednu odbočku, současně ale nevylučuje
skutečný překryv, který v pixelech vypadá jako Y. U rovnoběžek může metoda
vysoko ohodnotit dvě nespojené čáry. Pokud měl vstupní „klastr“ znamenat
jednu souvislou komponentu, je takový případ důvodem zkontrolovat seskupení
dat; dvě nespojené čáry samy o sobě nedokazují překryv. Nový sběrač proto
ukládá také počet 8-sousedních komponent.

**Kontrola na jednoduchých tvarech**

Toto jsou přesné výstupy nového testu, se seedem M = 42. Jsou to umělé
tvary definované v testech, nikoliv vybrané události z detektoru.

| Tvar | M | T | G |
|---|---:|---:|---:|
| Rovná čára | 0,00 | 0,00 | 0,00 |
| Souvislá široká čára | 0,00 | 0,00 | 0,00 |
| Ohyb L | 0,00 | 0,00 | 20,00 |
| Křížení X | 98,77 | 80,00 | 20,00 |
| Rozvětvení Y | 0,00 | 45,00 | 20,00 |
| Dvě oddělené rovnoběžky | 100,00 | 0,00 | 14,57 |

G dává u L, X i Y stejnou hodnotu 20, protože se na těchto tenkých
tvarech uplatňuje převážně nekonvexnost. Takové kritérium je samo o sobě
nerozlišuje. M je zde užitečné pro odlišný typ geometrie než T, ale teprve
reálné štítky ukážou, zda tuto výhodu využijeme na daném měření.

**Co bych nyní změnil v postupu**

1. Zmrazit dosavadní metody jako výchozí srovnání. Nepřepisovat současně
   M, T, G i jejich váhy: po změně více věcí by se těžko poznávalo, co pomohlo.
2. Ponechat T jako hlavní zdroj kandidátů; M ověřit na případech, které
   vysoké T nemají. G automaticky nepřidávat do hlavního výběru. Volitelně
   mu věnovat malou samostatnou kontrolní skupinu. Velikost zatím jen evidovat.
3. V každém měření označit pilotní dávku: 60 rovnoměrně náhodných klastrů,
   až 40 vysokých T a až 20 M mimo T. Je to zvládnutelný začátek pro
   odhalení slepých míst, nikoliv univerzální velikost dostatečného tréninku.
4. První rozhodnutí dělat nad původními pixely bez známého skóre. U
   nerozhodnutelných případů použít `uncertain`. Prohlédnutí skeletonu nebo
   modelu je možné, ale musí se evidovat pro odlišení slepé kontroly.
5. Porovnat, kolik jasných vizuálních pile-upů každá metoda navrhla, kolik
   dala falešných poplachů a které případy našla navíc k T. Současně
   prohlédnout pile-upy v náhodné kontrole, které skóre přehlédla.
6. Až podle konkrétních chyb vybrat další úpravu. Pokud M nepřidá nic,
   dočasně ho odstavit z hlavního filtru. Pokud pomůže hlavně přímým
   stopám, ponechat ho jako specializovaný doplněk. U G rozhodnout stejně.

Pracovní součet s váhami 40/30/20/10 vznikl jako hypotéza, nikoliv z dat.
Nedoporučuji nyní jen přeházet váhy podle dojmu. Všechny tři metody
čerpají ze stejné prostorové geometrie, G navíc využívá skeleton T.
Jejich hodnoty nejsou nezávislé důkazy a nemají společnou kalibraci.
Pro sběr lze použít logickou unii prahů nebo prosté `max(M,T)` pro řazení.
Číslo 80 pak stále znamená sílu pravidla, nikoliv 80% pravděpodobnost.

**Co by mělo přijít po pilotním označení**

Pro T2 dává smysl zkomprimovat skeleton na graf ramen: skutečné délky
po grafu v pixelových jednotkách, propojení konců s konkrétními uzly,
úhly mezi větvemi a možnost plynule spojovat ramena do více trajektorií.
Ověřit stabilitu při změně délky ořezu, otočení a šířky stopy. T2 pak
porovnat s T1 na stejných označených případech. Ani čtyři dlouhá ramena
nelze bez dalšího prohlásit za důkaz dvou částic.

Přímkové M případně rozšířit na cesty v grafu pro zakřivené elektronové
stopy, až chyby M2 ukážou, že to skutečně potřebujeme. Výsledek by měl
ukazovat konkrétní podpořené trajektorie a nevysvětlené pixely.

Pro první učený model je vhodný srovnávací Random Forest nad jednotlivými
vlastnostmi, nikoliv jen nad jedním ručně smíchaným skóre. Připravit
jasné kladné i záporné případy z více měření; `uncertain` nebrat jako
automaticky záporné. CNN porovnávat až se stabilními štítky a baseline.
Kolik dat bude stačit, určit z učicích křivek a množství různých typů
stop, ne z předem slíbeného čísla.

Trénink a test dělit přinejmenším po snímcích; pro přenos mezi měřeními
samostatně testovat celé dosud neviděné měření. Náhodné rozdělení klastrů
ze stejného snímku do obou částí může nadhodnotit zobecnění. U synteticky
skládaných překryvů musí být všechny varianty téhož původního tracku
v jedné části dělení. Překryvy lze použít pro doplňkový test, ale jejich
realismus závisí na odpovědi detektoru; prosté sečtení není automaticky
věrný model reálné akvizice.

Čas se nepoužívá. U pixelového signálu nejsou dosud všude ověřené jednotky
a kalibrace, takže tuto veličinu nelze rovnou vydávat za energii nebo
společný parametr napříč všemi třemi datasetovými typy. Přesně překryté
události mohou být z jediného prostorového obrazu nerozlišitelné od
jediné události; nejistota patří do výsledku i do ručních štítků.

**Technické zdroje pro návazné změny**

- Scikit-image, Skeletonize: https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html
  — co dělá ztenčení masky, rozdíl mezi skeletonize a medial axis, vliv
  pravidel na výslednou kostru. Skeleton není fyzikální rekonstrukce částice.
- Scikit-learn, GroupKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
  — dělení se skupinami, které se mezi tréninkem a testem nepřekrývají.

Implementace byla zkontrolována 11 testy v dodaném souboru. Reálná CLOG
měření a vlastní uložená CSV ručních štítků zde nebyla k dispozici;
výkonnost na těchto datech proto zatím není potvrzená.
