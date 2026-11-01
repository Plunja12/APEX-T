# APEX-T — uklizená pracovní verze

Hlavní příkazy z kořene projektu:

```powershell
python -m pip install -e .
python scripts/viewer.py --dataset 2
python scripts/collect_pileup_scores.py collect --dataset 2
python scripts/collect_pileup_scores.py report --dataset 2
python -m unittest discover -s tests -v
```

| Soubor | Účel |
|---|---|
| scripts/viewer.py | Původní prohlížeč snímků s tvými komentáři. |
| scripts/inspect_frames.py | Diagnostika prvních neprázdných snímků všech tří datasetů. |
| scripts/collect_pileup_scores.py | Hlavní společný sběr a ruční hodnocení M2/T1/G1. |
| scripts/label_clusters_m3.py | Zachovaný experimentální označovač M3 s diagnostikou signálu. Původně label_clusters.py. |
| src/pileup/io/clog_reader.py | Tvoje čtečka, obsah beze změny. |
| src/pileup/multiple_tracks.py | Tvoje M3.0-curves-signal, obsah beze změny. |
| src/pileup/topology.py | T1.1, obsah beze změny. |
| src/pileup/geometry.py | G1.0, obsah beze změny. |
| src/pileup/collection_baseline/ | Samostatně pojmenované zmrazené M2/T1/G1 pro srovnatelný sběr. |
| src/pileup/score_collection.py | Obnovený skutečný sběrač, ne kopie topologie. |

Sběrač záměrně měří M2, zatímco experimentální označovač používá M3.
Verze jsou oddělené, protože jejich skóre není přímo zaměnitelné.

`test_m_v2.py` nyní výslovně importuje M2 z collection_baseline;
`test_m_v3.py` importuje veřejné M3. Ostatní testy zůstávají v tests.
SciPy je doplněno do deklarovaných závislostí v pyproject.toml.
requirements.txt odkazuje na instalaci projektu pomocí -e .

Původní kód je po instalaci ve _backup_cleanup/<čas>/.
Staré označovače a chybně pojmenované kopie jsou tam zachované pro dohledání.
Nejsou v aktivní importní cestě ani hlavním adresáři scripts.
`plot_frame.py` patřil ke starší čtečce: importoval neexistující read_frames;
pro prohlížení používej viewer.py. explore.py byl prázdný.

Data, výsledky, štítky, konfigurace a .git se instalací nemění.
Staré štítky importuj podle návodu README_SCORE_COLLECTION_CZ.md v balíčku;
nový sběrač je nepovažuje automaticky za nově slepě označenou kontrolu.

Ověření v dostupném prostředí Python 3.12: 16 unittest testů, syntetická
ověření M2/M3/T spouštěná při načtení testů, import všech čtyř aktivních
skriptů a --help u tří skriptů s argumenty. Nový integrační test používá
přímo tvoji dodanou CLOG čtečku na umělém souboru. Reálná měření nebyla
součástí ZIPu. Instalátor nebyl spuštěn ve Windows; PowerShell tu není.
