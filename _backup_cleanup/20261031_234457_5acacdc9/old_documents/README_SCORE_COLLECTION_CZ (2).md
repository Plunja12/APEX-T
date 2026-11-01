**APEX-T — společný sběr M / T / G a ručních štítků**

Tento doplněk dokončuje sběr podkladů pro vyhodnocení skóre. Finální váhy
ani pravděpodobnost pile-upu zatím neurčuje. Obsahuje vlastní zmrazené
kopie M2.0, T1.1 a G1.0 z dosavadních ZIPů; původní metody, skripty,
měření a starší CSV nepřepisuje. Pokud jsi mezitím lokálně upravil M/T/G,
tento sběrač tyto lokální úpravy automaticky nepřebírá.

**Spuštění**

1. Rozbal obsah ZIPu do kořene svého existujícího projektu APEX-T, tedy
   tam, kde již jsou složky `scripts`, `src`, `data`. Není to celý projekt:
   používá tvůj stávající `src/pileup/io/clog_reader.py` a jeho `iter_frames`.
2. V PowerShellu otevři kořen projektu. Závislosti jsou stejné jako u G:

   ```powershell
   python -m pip install numpy scipy matplotlib
   ```

3. Spusť sběr na datasetu 2:

   ```powershell
   python scripts/collect_pileup_scores.py collect --dataset 2
   ```

   Nejprve se jednou spočítají metody nad celým souborem. Průběh se vypisuje
   do terminálu. První výpočet může trvat, zejména u velkých klastrů;
   okno se otevře až po dokončení. Další spuštění použije uložený index.
   Přerušení Ctrl+C ponechá dokončené snímky; další spuštění naváže.
   Částečný index nelze omylem použít pro náhodný kontrolní výběr.

   Jestli automatický výběr nenajde právě jeden soubor:

   ```powershell
   python scripts/collect_pileup_scores.py collect --dataset 2 --file "data/raw/cesta/k/mereni"
   ```

**Co dostaneš k hodnocení**

| Skupina | Výchozí počet | Účel |
|---|---:|---|
| Náhodná kontrola | 60 | Rovnoměrný náhodný výběr všech neprázdných vstupních klastrů; i drobných, chybových a s nízkými skóre. |
| Vysoké T | až 40 | T ≥ 40; pracovní kandidáti z topologie. |
| M mimo T | až 20 | M ≥ 40 a současně dostupné T < 40; test přínosu M navíc. |
| G mimo M a T | 0 | Volitelný test; výchozí výběr G nepoužívá. |
| Největší klastry | 0 | Volitelný průzkum velikosti, bez uměle zavedeného skóre A. |

Kontrola se vylosuje jako první z celého souboru; nevybírá se pouze z nízkých
skóre. Skupiny se v jedné dávce nepřekrývají a promíchají se. Nedostatek
kandidátů v některé skupině se nedoplňuje jiným typem klastrů. Prázdné
klastry se evidují počtem a nevstupují do výběru.

**Ovládání**

- `1` nebo první tlačítko: `pileup`.
- `2`: `not_pileup`.
- `3`: `uncertain` — z dostupných pixelů neumíš rozhodnout. Nejistý případ
  nemusíš převádět na negativní příklad.
- `D`: ukázat/skrýt T skeleton, přímkové hypotézy M, hodnoty skóre a původ
  kandidáta. Zobrazení se trvale zaznamená; takový štítek již nevstoupí do
  metrik nové slepé kontroly. Skrytí zpět tento záznam neruší.
- Mezerník: přeskočení jen v aktuální relaci, bez uložení štítku. Při
  dalším otevření dávky se neoznačené klastry nabídnou znovu.
- `Q`: zavření. Označené klastry se při obnovení vynechají.
- Do `Notes` můžeš psát vysvětlení. Při psaní poznámek číslice neukládají štítek.

Vlevo nahoře je celý snímek s rámečkem, vpravo původní klastr bez prahování
signálu. Detail má vlastní barevnou škálu pro čitelnost; celé měření se
neřeže podle intenzity. Šedé aktivní pixely znamenají chybějící konečný signál.
Barva je neověřený pixelový signál, nikoliv automaticky energie v keV.
Čas a čtvrtá položka pixelového záznamu se do metod nepředávají.

Po `D` jsou vidět souvislé čáry skeletonu a koncové/rozvětvovací body.
M kreslí odhadnuté úsečky v rozsahu podpůrných pixelů. Jsou to modelové
hypotézy, nikoliv hotová separace nebo ověřený počet částic. G se číselně
ukáže v textu; nepřekrývá původní obraz dalšími barevnými tečkami.

**Uložení a vyhodnocení**

V `results/collection/` vzniká samostatná databáze a exporty:

- `scores.sqlite3`: průběžně potvrzované výsledky, původní x/y/signál,
  diagnostika, pevné výběrové dávky a ruční štítky.
- `<davka>_scores.csv`: skóre a vlastnosti všech neprázdných klastrů
  aktuálního měření; štítky jsou připojené, pokud existují.
- `<davka>_labels.csv`: označené klastry. Aktualizuje se po každém štítku;
  celý výpočet se při kliknutí neopakuje.
- `<davka>_report.json`: počty podle skupin, nejisté/chybějící štítky,
  přesnost výběru a záchyt vůči vizuálním štítkům na slepé kontrole,
  přínos M/G navíc a počty abstencí.
- `<beh>_methods.json`: identita měření a přesné verze, výchozí parametry
  a otisky zdrojového kódu metod/čtečky.

Export po ukončení nebo kdykoliv později:

```powershell
python scripts/collect_pileup_scores.py report --dataset 2
```

Pokud existuje více dávek, příkaz vypíše jejich identifikátory; vyber jednu:

```powershell
python scripts/collect_pileup_scores.py label --dataset 2 --batch ZACATEK_ID
python scripts/collect_pileup_scores.py report --dataset 2 --batch ZACATEK_ID
```

Stejná data, verze metod, seed a nastavení obnovují stejnou pevnou dávku.
Pro další průzkumnou dávku zvol nový seed. Náhodná kontrola se stále losuje
z celé populace, takže může obsahovat starší štítky. Jejich původ zůstane
zachován a nepovažují se za nově nezávisle hodnocenou slepou kontrolu.

```powershell
python scripts/collect_pileup_scores.py collect --dataset 2 --seed 2027
```

Volitelný, explicitní test zbývajících hypotéz:

```powershell
python scripts/collect_pileup_scores.py collect --dataset 2 --g-samples 20 --size-samples 20 --seed 2028
```

Další měření spusť stejně s `--dataset 1` a `--dataset 3`. Index nemá
výchozí filtr 16 pixelů: malá místa zůstanou v kontrolní populaci, ale
nepoužitelná skóre budou prázdná se stavem `not_applicable`. Chyba metody
je prázdné skóre s vysvětlením, nikoliv umělá nula.

Pracovní sloupec `S_MT = max(M,T)` slouží pouze k řazení a nedělá z metod
nezávislé důkazy. `S_MT_status=partial` označuje, že je dostupná jen jedna
složka. Zpráva vyhodnocuje unii při samostatných prahových hodnotách M/T.
`size_percentile` je empirický percentil počtu pixelů v daném měření,
s průměrným pořadím shodných velikostí. Není to pravděpodobnost ani hotové A.

Neúplné, nejisté a odtajněné kontroly jsou ve zprávě výslovně viditelné.
`screening_recall` počítá kladný případ bez použitelného skóre mezi
nezachycené; chybové skóre přesto není uloženo jako negativní hodnota.
Intervaly ve zprávě jsou orientační Wilsonovy intervaly na úrovni klastrů;
nezohledňují závislost klastrů ze stejného snímku. Fyzikálně nerozhodnutelné
pile-upy tímto neodhalíme a vizuální štítky nejsou fyzikální ground truth.
Kontrolu, podle které vybereš metody nebo naladíš prahy, již pro konečné
hodnocení modelu považuj za vývojová data. Konečný test musí být nový.

**Převzetí starších ručních štítků**

Nejprve vytvoř dávku bez otevření okna:

```powershell
python scripts/collect_pileup_scores.py sample --dataset 2
python scripts/collect_pileup_scores.py import --dataset 2 --csv results/labels/dataset_2_mt.csv
python scripts/collect_pileup_scores.py label --dataset 2
```

Import vyžaduje přesnou shodu sloupce `source`. Při přesunu projektu mezi
počítači můžeš výslovně určit starou cestu pomocí `--legacy-source`.
Tím potvrzuješ přiřazení k aktuálnímu měření; staré CSV nemá otisk obsahu
a automaticky nelze dokázat totožnost původních dat. Skript ověřuje ID
snímku/klastru a počet pixelů, pokud je v CSV. Nikdy nehádá soubor podle
stejného názvu. Konfliktní štítky zruší celý import, bez částečného zápisu.
Importované štítky jsou označeny jako starší a nevstoupí do nové slepé
kontroly. Zdrojová CSV se nepřepisují.

**Ověření a co zatím ověřeno není**

```powershell
python -m unittest discover -s tests -p test_score_collection.py -v
```

Testy používají syntetické tvary a testovací adaptér rozhraní `iter_frames`,
nikoli tvé skutečné CLOG soubory. Ověřují izolaci časů a signálu, vynechání
prázdných klastrů se zachováním ID, chyby místo falešných nul, obnovu/cache,
pevný náhodný výběr, původ štítků, atomický import a zobrazení bez přepočtů.
Vytvoření indexu nad reálnými měřeními a jejich správné fyzikální hodnocení
je třeba provést u tebe. Nemáme zatím naměřenou přesnost M/T/G.

Pokud CSV drží otevřený Excel a Windows zabrání přepsání, štítek zůstane
v databázi. Zavři CSV a znovu spusť `report`.
