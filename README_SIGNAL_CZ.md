**APEX-T — integrace pixelového signálu, verze collection-1.1-signal**

Tato aktualizace navazuje na uklizenou verzi se skripty viewer.py,
inspect_frames.py a collect_pileup_scores.py. Nevrací odstraněné staré
označovače ani veřejné M3/T/G moduly. V ZIPu je přesně pět souborů:

- `src/pileup/signal_features.py`: nový modul signálových vlastností a E1.
- `src/pileup/score_collection.py`: aktualizace sběrače.
- `scripts/collect_pileup_scores.py`: aktualizace stejného spouštěcího skriptu.
- `tests/test_signal_features.py`: nové testy.
- `README_SIGNAL_CZ.md`: tento návod.

**Instalace**

Vlož APEX-T-signal-update.zip do kořene stávajícího projektu a z tohoto
kořene spusť PowerShell příkaz:

```powershell
Expand-Archive -LiteralPath .\APEX-T-signal-update.zip -DestinationPath . -Force
python scripts/collect_pileup_scores.py collect --dataset 2
```

Rozbal celý ZIP se složkami. Jednotlivé soubory nepřetahuj do jedné složky
a nepřejmenovávej je. Aktualizace používá dosavadní závislosti: NumPy,
SciPy a Matplotlib. Není zde další instalátor ani nový označovací program.

**Co je signál a co je energie**

Do výpočtu vstupuje třetí hodnota CLOG pixelu, kterou čtečka ukládá do
`value`. Čtvrté pole `extra` ani čas se nepoužívají. Kód neprovádí
kalibraci. Výchozí jednotka je `unknown`. ToT nelze automaticky pokládat
za keV ani jeho prostý součet za uloženou energii.

Pokud je potvrzeno, že soubor už obsahuje kalibrované energie, můžeš
jednotku výslovně zapsat do metadat:

```powershell
python scripts/collect_pileup_scores.py collect --dataset 2 --signal-unit keV
```

`--signal-unit` nic nepřepočítává a nedokazuje platnost kalibrace. U
nekalibrovaného ToT můžeš uvést `--signal-unit ToT`. Pokud jde o počet
impulsů, neoznačuj jej za energii; zatím ponech unknown a ověř režim exportu.

**Nové uložené vlastnosti**

| CSV sloupec | Význam |
|---|---|
| signal_sum | Součet všech hodnot klastru; při chybějících hodnotách prázdný. |
| signal_observed_sum | Součet dostupných konečných hodnot; u neúplného klastru pouze diagnostika. |
| signal_mean | Součet / počet aktivních pixelů. Není to dE/dx. |
| signal_median, signal_max, signal_p90 | Medián, maximum a 90. percentil pixelových hodnot. |
| signal_cv | Směrodatná odchylka / průměr: proměnlivost hodnot bez jednotek. |
| signal_max_to_median | Dominance nejsilnějšího pixelu vůči mediánu. |
| signal_top10_fraction | Podíl celkového signálu v nejsilnějších ceil(0.1*N) pixelech. |
| signal_centroid_shift_px | Vzdálenost signálově váženého těžiště od geometrického těžiště. |
| signal_centroid_shift_norm | Stejný posun dělený úhlopříčkou souřadnicového rozsahu, nejméně 1 pixel. |
| signal_peak_neighbor_ratio | Maximum / medián ostatních aktivních pixelů do vzdálenosti 2 v Chebyshevově metrice; nejméně tři sousedé. |
| signal_junction_ratio | Největší poměr mediánu v okolí uzlu T k mediánu na vzdálenější kostře. |
| touches_detector_edge | Klastr obsahuje pixel na hranici 0 nebo 255. Signál může být oříznutý detektorem. |
| signal_status | Úplnost, záporné hodnoty, nulový součet, překročený zadaný limit apod. |
| E, E_status, e_reference_n, e_mean_percentile | Pracovní signálové skóre, jeho dostupnost a popis reference. |

Poměr u uzlu používá aktivní pixely do vzdálenosti 2 od reprezentanta
uzlu T; referenční hodnoty bere z bodů kostry vzdálených alespoň 4 od
všech těchto uzlů. Potřebuje nejméně tři hodnoty v okolí a pět na ramenech.
Reprezentant uzlu je přibližný: nejde o fit fyzikálního průsečíku.

Chybějící/nekonečné hodnoty nedostanou umělý nulový signál. Záporné nebo
zcela nulové hodnoty se evidují, ale nevstupují do E. Poměry s nulovým
jmenovatelem jsou prázdné. Signál není prahován za účelem vytvoření nové
geometrie: M2/T1/G1 a jejich význam se nemění.

**Jak funguje E1**

E je záměrně jednoduchá, explicitní hypotéza k ručnímu ověření:
„Má tento klastr nezvykle vysoký průměrný signál vzhledem k podobnému počtu pixelů?“

1. Reference vzniká zvlášť pro každý vstupní soubor. Měření se nemíchají.
2. Velikosti se rozdělí do skupin N = 1; 2–3; 4–7; 8–15; 16–31; 32–63 atd.
3. V každé skupině se seřadí průměrný signál úplných, nezáporných klastrů
   s kladným součtem a bez překročení výslovně zadaného limitu.
4. Pro každou hodnotu se spočítá empirický percentil p v rozsahu 0–1.
   Shody dostanou prostřední pořadí; vlastní případ je v referenci zahrnutý.
5. E = 100 * clip((p - 0.90) / 0.10, 0, 1).

E >= 50 tedy znamená přibližně horních 5 % průměrných signálů v příslušné
velikostní skupině. Při shodách nemusí být nad prahem právě 5 % případů.
Pokud mají všechny klastry stejný průměr, dostanou E = 0. Skupina musí mít
výchozím nastavením alespoň 30 použitelných klastrů; jinak zůstane E prázdné
a E_status = insufficient_reference. Počet lze změnit přes --e-min-reference.

Jde o popis pořadí v měření, nikoliv o referenci ověřených jednotlivých
částic. Reference může obsahovat pile-upy. Seskupení podle N neodstraní
vliv typu částice, úhlu, šířky stopy nebo všech změn nastavení detektoru.
Vysoké E může dostat normální jednotlivá částice; naopak překryv nemusí
mít mimořádný průměr. Poměrové vlastnosti jsou invariantní vůči společnému
kladnému násobku všech hodnot, nikoliv vůči nelineární či pixelově různé odezvě.

Ostatní signálové vlastnosti nejsou skrytě smíchány do E. Ukládají se pro
diagnostiku a budoucí model. S_MT zůstává max(M,T). E se nevydává za
pravděpodobnost ani za fyzikální důkaz pile-upu.

**Výběr a ovládání**

Výchozí dávka obsahuje 60 rovnoměrně náhodných klastrů, až 40 vysokých T,
až 20 vysokých M mimo T a nově až 20 případů E >= 50 s dostupnými M < 40
a T < 40. Skupina E_without_MT tedy testuje přírůstek oproti geometrii.
Protože vyžaduje naměřené M/T, nenabírá klastry menší než 16 pixelů;
ty však nadále patří do náhodné kontroly a E se pro ně může ukládat.
G se standardně nepoužívá k výběru. Skupiny zůstávají disjunktní a při
nedostatku kandidátů se nenaplňují jinými případy.

- 1 / 2 / 3: pileup / not_pileup / uncertain.
- D: zobrazení geometrie nebo její skrytí; z E pohledu návrat ke geometrii.
- E: zobrazení histogramu signálu, E, počtu referencí a souhrnných hodnot;
  další E přepne zpět na geometrii, D pak umožní vše skrýt.
- Mezerník: přeskočení bez štítku v aktuální relaci. Q: zavření.

E i D trvale zaznamenají prohlédnutí diagnostiky. Takový štítek nepatří
do nově slepě hodnocené kontroly. Při klikání se metody znovu nepočítají.

Pokud chceš prohlédnout především novou větev:

```powershell
python scripts/collect_pileup_scores.py collect --dataset 2 --control 30 --t-samples 0 --m-samples 0 --e-samples 30 --seed 2031
```

Při malém počtu vzácných E kandidátů jich může být méně než požadovaných 30.
Zvýšení součtu signálu se nikdy nepoužije jako automatický ruční štítek.

**Cache, staré štítky a report**

Data měření, databáze a staré dávky se nemažou. Aktualizace vytvoří nový
verzovaný běh/dávku. Štítky jsou připojené k otisku zdrojového souboru a
ID klastru, takže se zachovají a již označené kandidáty nemusíš opakovat.
Při shodě otisků známého starého sběrače, geometrických metod, čtečky i
číselných knihoven převezme M/T/G z cache a dopočítá pouze signál/E.
Pokud shodu nelze ověřit, geometrii přepočítá. Nové indexy jsou označené
jako dokončené teprve po sestavení celé signálové reference.

Nová dávka může v náhodné kontrole obsahovat již označené případy. Zůstanou
v kontrolním výběru, ale nepovažují se za nové slepé anotace. Program je
nenahrazuje účelově jinými případy. To může zmenšit novou kontrolu a report
tuto okolnost uvádí. Historické dávky se nevyhodnocují jako by měly E.

Report poslední vytvořené dávky:

```powershell
python scripts/collect_pileup_scores.py report --dataset 2 --batch latest
```

`latest` je výslovná volba poslední vytvořené dávky; může zahrnovat jiné
výběrové nastavení, pokud jsi ho použil. Přesné ID dávky je ve výpisu a
lze jej předat místo latest. Report porovnává M, T, G, E, unii M/T a unii
M/T/E na stejných dostupných slepých kontrolních štítcích. Uvádí přírůstek
E nad M/T, neaplikovatelná skóre a problémy signálu. Neúplná či nejistá
kontrola neprokazuje přesnost na všech fyzických pile-upech.

**Saturace a další okolnosti**

Pokud znáš limit elektroniky/exportu ve stejných jednotkách, můžeš uvést
`--signal-ceiling HODNOTA`. Pixely na/za limitem se označí a daný klastr
nevstoupí do E ani jeho reference. Limit se neodhaduje z maxima datasetu.
Samotná shoda s limitem neprokazuje saturaci a bez známého limitu ji kód
netvrdí. Složitější zkreslení odezvy tímto příznakem odhalit nemusíme.

Pro další kolo mají smysl: profil signálu podél kostry a oddělená maxima,
délka a šířka stopy, kontrola okraje a vadných pixelů, nastavení měření
a zaplnění snímku. Skutečné dE/dx vyžaduje kalibrovanou energii a délku
dráhy v senzoru, nikoliv pouze počet pixelů nebo délku projekce.

Technické podklady:

- J. Jakubek, Precise energy calibration of pixel detector working in time-over-threshold mode:
  https://doi.org/10.1016/j.nima.2010.06.183 — per-pixel převod ToT a jeho nelinearita.
- M. Sommer, C. Granja, S. Kodaira, O. Ploc, High-energy per-pixel calibration of Timepix pixel detector with laboratory alpha source:
  https://doi.org/10.1016/j.nima.2021.165957 — vliv odezvy a biasu, sdílení náboje a vysokých signálů.

**Ověření**

```powershell
python -m unittest discover -s tests -v
```

Ověřena byla uklizená struktura bez odstraněných veřejných M/T/G modulů.
Testy pokrývají nezměněnou geometrii, časovou nezávislost, reference a shody,
škálování signálu, lokální poměr u uzlu, chyby a limit signálu, převzetí
cache bez ztráty štítků, neznámou verzi cache, report, skutečnou CLOG
čtečku na umělém vstupu a GUI bez přepočítávání metod.
Účinnost E na tvých skutečných měřeních zde ověřena nebyla.
