
import argparse  # zpracovávání terminálu
from pathlib import Path  # cesty souborů
import matplotlib.pyplot as plt  # grafické okno
import matplotlib.patheffects as pe  # efekty
import numpy as np  #matice

from matplotlib.colors import Normalize  # barevná škála
from matplotlib.patches import Rectangle  # rámeček kolem clusteru

from pileup.io.clog_reader import (
    iter_frames, 
    get_frame,
    frame_to_matrix,
    cluster_to_local_matrix
)
# práce se souborem/ iterace, frames, matice, clustery

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
# přizbůsobení cest v operačním systému

def find_data_files():
    """Find likely measurement files in data/raw."""

    files = [] #prázdný seznam files, na cesty k nalezeným souborům

    for path in DATA_DIR.rglob("*"): #data/raw, hledá položky, uloží do path
        if not path.is_file(): 
            continue #pokud není soubor, přeskočí ji

        name = path.name.lower()
        extension = path.suffix.lower() #ve všech variantách písmen m/V

        if extension in {".zip", ".png", ".jpg", ".jpeg"}:
            continue

        if extension == ".clog" or name.startswith(
            ("tot_", "itot", "cnt_")
        ): 
            files.append(path)
#pokud platí koncovka .clog a název je tot, itot nebo cnt_ zapíše se na konec files []
### najít obecné řešení později
    return sorted(files)
#předá hotový seznam

def select_file(dataset, custom_file=None): #vybírá soubory z datasetu, možno zadat parametr
    """Select a measurement file."""

    if custom_file is not None:
        path = Path(custom_file)

        if not path.is_absolute(): #C:\Projects\APEX-T\data\raw\ = absolutní cesta
            path = PROJECT_ROOT / path

        if not path.is_file():
            raise FileNotFoundError(path)

        return path
#pokud soubor existuje, vrátíme jeho cestu
    prefix = f"{dataset:02d} "
# prefix ve formě "01 ", píšeme 1/2/3..

    matches = [
        path for path in find_data_files()
        if path.relative_to(DATA_DIR).parts[0].startswith(prefix)
    ]
#tvorba seznamu, vezme soubory a nechá jen ty co patří daného datasetu
    if not matches:
        raise FileNotFoundError(
            f"No measurement file found for dataset {dataset}. "
            "Use --list to inspect available files."
        )
#žádný výsledek error
    if len(matches) > 1:
        raise ValueError(
            "Multiple files found. Select one with --file:\n"
            + "\n".join(str(path) for path in matches)
        )
#více výsledků error
    return matches[0]

def print_frame_stats(source, limit=200): ###kontolujeme jen 200 frames, ne celý snímek
    """Analyze pixel and cluster counts in the first frames."""

    stats = []  #záznam pro každý snímek
    total_clusters = 0  #součet clusters
    total_pixels = 0  #součet pixelů
    total_duplicates = 0  #opakované záznamy souřadnic

    for frame_id, clusters in iter_frames(source):  #funkce z clog. reader

        pixel_count = sum(len(cluster) for cluster in clusters) #součet pix. každého clusteru
        cluster_count = len(clusters) 
### nasčítání clusterů není pravdivá hodnota kvůli pileups
        coordinates = [
            (pixel["x"], pixel["y"])
            for cluster in clusters
            for pixel in cluster
        ]

        unique_pixels = len(set(coordinates))
        duplicates = pixel_count - unique_pixels
###kontrola duplicity, bude potřeba i později ???
        stats.append((frame_id, cluster_count, pixel_count, duplicates))

        total_clusters += cluster_count
        total_pixels += pixel_count
        total_duplicates += duplicates

        if len(stats) >= limit:
            break
#hodnota říká kolik snímků jsme ctěli prozkoumat, ukončí cyklus
    print(f"\nFile: {source.name}")
    print(f"Frames analyzed: {len(stats)}")
    print(f"Total clusters: {total_clusters}")
    print(f"Total pixel records: {total_pixels}")
    print(f"Duplicate coordinates: {total_duplicates}")

    print("\nFive frames with most pixel records:")

    densest = sorted(
        stats,  #nový seznam
        key=lambda item: item[2],  #pixel count, v naší čtveřici, lambda je prázdná fuknkce
        reverse=True
    )[:5]

    for frame_id, clusters, pixels, duplicates in densest:
        print(
            f"Frame {frame_id}: "
            f"{clusters} clusters, "
            f"{pixels} pixels, "
            f"{duplicates} duplicates"
        )
#nemá return ale vypíše se do terminálu

def create_viewer(source, frame_id, first_cluster):
    """Create an interactive full-frame and cluster viewer."""

    if frame_id is None:
#hledáme none, 0 by mohla být platné číslo snímku
        for candidate_id, candidate_clusters in iter_frames(source):

            if any(candidate_clusters):
                frame_id = candidate_id
                clusters = candidate_clusters
                break
#obsahuje alespoň jeden cluster a pixely ?
        else:
            raise ValueError("No non-empty frames found")

    else:
        # Load a specific frame when requested.
        clusters = get_frame(source, frame_id)

    if not clusters:
        raise ValueError("Selected frame has no clusters")
#kontrola při ručním výběru
    print(f"Selected frame: {frame_id}")

    frame_matrix = frame_to_matrix(clusters)
#funkce z clog_reader = převede na 256 x 256
    pixel_lookup = {}
#slovník
    for cluster_id, cluster in enumerate(clusters):
        for pixel in cluster:
            pixel_lookup[(pixel["x"], pixel["y"])] = cluster_id
#enumerate hodí obsah i číslo položky
    values = frame_matrix[np.isfinite(frame_matrix)]

    if len(values) == 0:
        raise ValueError("No valid pixel values")

    vmin = float(values.min())
    vmax = float(values.max())
#příprava pro barevnou škálu
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5
#kdyby byly hodnoty moc podobné
    norm = Normalize(vmin=vmin, vmax=vmax)

    cmap = plt.get_cmap("viridis").copy()  #mapa viridis
    cmap.set_bad("#f0f0f0")  #barva pro nesasaženou část detektoru

    fig, (ax_full, ax_detail) = plt.subplots(
        1, 2,  #řádek, sloupce 
        figsize=(15, 7),  #15 x 7 palců
        constrained_layout=True  #automatické rozvržení místa pro grafy, barevná stupnice a popisky
    )

#tvorba celého obrazu pro graf
    full_image = ax_full.imshow(
        frame_matrix,
        origin="lower",  #y roste nahoru
        cmap=cmap,  #naše mapa
        norm=norm,  #škála
        interpolation="nearest"  #nejbližší pixeů
    )

    ax_full.set_title(f"Full detector — frame {frame_id}")
    ax_full.set_xlabel("X pixel")
    ax_full.set_ylabel("Y pixel")

    fig.colorbar(
        full_image,
        ax=[ax_full, ax_detail],
        label="Energy",  ###nutno upravit dle potřeby na Kev / Mev
        shrink=0.8
    )

    selected = [0]
    highlight = [None]
#tím že je definovaná uvnitř create_viewer 
    def show_cluster(cluster_id):
        """Update the detail plot and highlight the selected area."""

        cluster = clusters[cluster_id]
#vybere cluster
        if not cluster:
            print(f"Cluster {cluster_id} is empty")
            return

        selected[0] = cluster_id
#zapamatuje si cluster který je zrovna vybraný
        local_matrix, xmin, ymin, xmax, ymax = (
            cluster_to_local_matrix(cluster)
        )
#malá matici přiblíženého clusteru, podle max a min
        ax_detail.clear()
#vyčistíme graf
        ax_detail.imshow(
            local_matrix,
            origin="lower",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            extent=(
                xmin - 0.5, xmax + 0.5,
                ymin - 0.5, ymax + 0.5
            )
        )
###stačí to odsazení 1?
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        fontsize = max(5, min(11, 140 / max(width, height)))
#úprava velikosti písma podle clusteru 5/11, 140 je konstanta

        for pixel in cluster:
            ax_detail.text(
                pixel["x"],
                pixel["y"],
                f'{pixel["value"]:g}', #:g je úsporný zápis
                ha="center",
                va="center",
                fontsize=fontsize,
                color="white",
                path_effects=[
                    pe.withStroke(
                        linewidth=2,
                        foreground="black"
                    )
                ]
            )
#nastavení grafu a písma uvnitř
        if width <= 20:
            ax_detail.set_xticks(range(xmin, xmax + 1))

        if height <= 20:
            ax_detail.set_yticks(range(ymin, ymax + 1))

        if width <= 40 and height <= 40:
            ax_detail.set_xticks(
                np.arange(xmin - 0.5, xmax + 1, 1),
                minor=True
            )
            ax_detail.set_yticks(
                np.arange(ymin - 0.5, ymax + 1, 1),   #posloupnost která vykreslí mřížku
                minor=True
            )
            ax_detail.grid(which="minor", alpha=0.3)  #mřížka aplha je průhlednost

        pixel_values = np.array(
            [pixel["value"] for pixel in cluster]
        )
#vezme z každého pixelu jeho hodnotu
        total = pixel_values.sum()
        minimum = pixel_values.min()
        maximum = pixel_values.max()

        ax_detail.set_title(
            f"Cluster {cluster_id} | "
            f"N={len(cluster)} | "
            f"Sum={total:g}\n"
            f"Min={minimum:g} | Max={maximum:g}"
        )
#popis jednoho clusteru
        ax_detail.set_xlabel("X pixel")
        ax_detail.set_ylabel("Y pixel")
#znovu popsat osy, clear je vymazal
        if highlight[0] is not None:
            highlight[0].remove()
#pokud rámeček exiatuje tak remove, aby se nehromadily
        rectangle = Rectangle(
            (xmin - 0.5, ymin - 0.5),
            width,
            height,
            fill=False,
            edgecolor="red",
            linewidth=1.8
        )
#obdélníková oblast kde bude cluster
        ax_full.add_patch(rectangle)
        highlight[0] = rectangle
#vykreslíme vytvořený obdélník, 
        fig.canvas.draw_idle()
###pomálé pro obrovské cluster, možná i ne moc přehledné, 
    def on_click(event):
        """Select a cluster by clicking a hit pixel."""

        if event.inaxes != ax_full:
            return
#kde nastalo kliknutí
        if event.xdata is None or event.ydata is None:
            return

        x = round(event.xdata)
        y = round(event.ydata)
#nemusím kliknout přímo na pixel, proto zaokrouhlím

        cluster_id = pixel_lookup.get((x, y))  #slovník, když kliknu mimo, nehodí chybu

        if cluster_id is not None:
            show_cluster(cluster_id)

    def save_view():
        """Save the current visualization."""

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
#výstupní složka, případně vytvoří i chybějící nadřazené složky
        filename = (
            f"{source.stem}_frame_{frame_id}_"
            f"cluster_{selected[0]}.png"
        )
#itot_frame_17_cluster_7.png
        destination = RESULTS_DIR / filename

        fig.savefig(destination, dpi=200)

        print(f"Saved: {destination}")
#uložíme soubor a napíšeme kam se uložil
    def on_key(event):
        """Navigate between clusters or save the figure."""

        if event.key == "right":
            next_id = (selected[0] + 1) % len(clusters)
            show_cluster(next_id)
#šipka doprava, zobrazí další cluster
        elif event.key == "left":
            previous_id = (selected[0] - 1) % len(clusters)
            show_cluster(previous_id)
#doleva podobně
        elif event.key == "p":
            save_view()
#p ukládá snímek
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
#funkce bez závorek, čeká na zavolání
    if not 0 <= first_cluster < len(clusters):
        raise ValueError(
            f"Cluster must be between 0 and {len(clusters)-1}"
        )
#povolený rozsah clusterů
    show_cluster(first_cluster)

    print(f"File: {source}")
    print(f"Frame: {frame_id}")
    print(f"Clusters: {len(clusters)}")
    print(f"Pixel records: {sum(len(c) for c in clusters)}")
    print("Click a hit pixel to select its cluster.")
    print("Left/Right: change cluster | P: save PNG")
#info o aktuálním snímku

   # save_view()
    plt.show()
###obrázek se uloží při startu, je to dobré ??

def main():
    """Read command-line arguments and start the viewer."""

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=int, default=1)  #číslo datasetu
    parser.add_argument("--frame", type=int, default=None)  #číslo snímku
    parser.add_argument("--cluster", type=int, default=0)  #cluster
    parser.add_argument("--file", type=str)  #vlastní cesta k souboru
    parser.add_argument("--list", action="store_true")  #výpis dostupných souborů
    parser.add_argument("--stats", action="store_true")  #kontorla

    args = parser.parse_args()  #čtení z terminálu

    if args.list:
        files = find_data_files()

        if not files:
            print("No measurement files found")

        for path in files:
            print(path.relative_to(PROJECT_ROOT))

        return

    source = select_file(args.dataset, args.file)

    if args.stats:
        print_frame_stats(source)
        return

    create_viewer(
        source=source,
        frame_id=args.frame,
        first_cluster=args.cluster
    )
#otevře prohlížeč

if __name__ == "__main__":
    main()
