import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk
import os

from config import CANDIDATE_COUNT, MULTIPLE_CANDIDATE_COUNT, THUMB_SIZE, TOP_RESULT_SIZE, COLOR_WEIGHT, EMBEDDING_WEIGHT, HASH_WEIGHT
from view.view_repository import calculate_similarities_for_reference, load_display_image

class SimilarityViewer:
    def __init__(self, root, repo, embedding_index):
        self.root = root
        self.repo = repo
        self.embedding_index = embedding_index

        self.page_size = 15
        self.current_page = 0

        self.total_images = self.repo.get_image_count()
        self.total_pages = (
            self.total_images + self.page_size - 1
        ) // self.page_size

        self.images = []
        self.image_by_id = {}
        self.photo_refs = {}
        self.selected_id = None
        self.selected_reference_ids = set()
        self.displayed_reference_ids = []
        self.current_results = []
        self.sort_column = "overall"
        self.sort_descending = True
        
        self.color_weight = tk.DoubleVar(value=COLOR_WEIGHT)
        self.embedding_weight = tk.DoubleVar(value=EMBEDDING_WEIGHT)
        self.hash_weight = tk.DoubleVar(value=HASH_WEIGHT)

        root.title("Image Similarity Explorer")
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        root.geometry(f"{screen_width}x{screen_height}+0+0")

        outer = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(outer, padding=10)
        right = ttk.Frame(outer, padding=10)
        outer.add(left, weight=1)
        outer.add(right, weight=2)
        
        self.reference_label = ttk.Label(
            right,
            text="Noch kein Referenzbild ausgewählt",
            font=("TkDefaultFont", 14, "bold")
        )
        self.reference_label.pack(anchor="w")

        self.reference_score = ttk.Label(
            right,
            text=""
        )
        self.reference_score.pack(anchor="w", pady=(2, 8))


        ttk.Label(
            right,
            text="Top 5 ähnliche Bilder",
            font=("TkDefaultFont", 12, "bold")
        ).pack(anchor="w", pady=(0, 6))

        self._build_weight_controls(right)

        self.top5_frame = ttk.Frame(right)

        self.top5_frame.pack(
            fill=tk.X,
            expand=False,
            pady=(0, 10)
        )

        self.top5_refs = []

        ttk.Label(left, text="Referenzbild auswählen", font=("TkDefaultFont", 14, "bold")).pack(anchor="w")
        ttk.Label(left, text="Klick auf ein Bild → rechts werden die ähnlichsten Bilder angezeigt.").pack(anchor="w", pady=(2, 10))

        page_navigation = ttk.Frame(left)
        page_navigation.pack(
            anchor="center",
            pady=(0, 5)
        )

        ttk.Label(
            page_navigation,
            text="Seite"
        ).pack(side=tk.LEFT)

        self.page_entry = ttk.Entry(
            page_navigation,
            width=7,
            justify="center"
        )
        self.page_entry.pack(
            side=tk.LEFT,
            padx=5
        )

        self.page_total_label = ttk.Label(
            page_navigation,
            text=""
        )
        self.page_total_label.pack(side=tk.LEFT)

        self.page_entry.bind(
            "<Return>",
            self._go_to_entered_page
        )

        navigation = ttk.Frame(left)
        navigation.pack(
            fill=tk.X,
            pady=(0, 5)
        )

        self.previous_1000_button = ttk.Button(
            navigation,
            text="← 1000",
            command=lambda: self.change_page(-1000)
        )
        self.previous_1000_button.pack(side=tk.LEFT)

        self.previous_100_button = ttk.Button(
            navigation,
            text="← 100",
            command=lambda: self.change_page(-100)
        )
        self.previous_100_button.pack(side=tk.LEFT)

        self.previous_10_button = ttk.Button(
            navigation,
            text="← 10",
            command=lambda: self.change_page(-10)
        )
        self.previous_10_button.pack(side=tk.LEFT)

        self.previous_button = ttk.Button(
            navigation,
            text="← 1",
            command=lambda: self.change_page(-1)
        )
        self.previous_button.pack(side=tk.LEFT)

        self.next_1000_button = ttk.Button(
            navigation,
            text="1000 →",
            command=lambda: self.change_page(1000)
        )
        self.next_1000_button.pack(side=tk.RIGHT)

        self.next_100_button = ttk.Button(
            navigation,
            text="100 →",
            command=lambda: self.change_page(100)
        )
        self.next_100_button.pack(side=tk.RIGHT)

        self.next_10_button = ttk.Button(
            navigation,
            text="10 →",
            command=lambda: self.change_page(10)
        )
        self.next_10_button.pack(side=tk.RIGHT)
        
        self.next_button = ttk.Button(
            navigation,
            text="1 →",
            command=lambda: self.change_page(1)
        )
        self.next_button.pack(side=tk.RIGHT)

        self.multi_selection_frame = ttk.Frame(left)

        self.multi_selection_label = ttk.Label(
            self.multi_selection_frame,
            text=""
        )

        self.multi_selection_label.pack(
            side=tk.LEFT,
            padx=(0, 10)
        )

        self.confirm_selection_button = ttk.Button(
            self.multi_selection_frame,
            text="Auswahl bestätigen",
            command=self.confirm_multi_selection
        )

        self.confirm_selection_button.pack(
            side=tk.LEFT
        )

        self.multi_selection_frame.pack_forget()

        self.thumb_frame = ttk.Frame(left)
        self.thumb_frame.pack(
            fill=tk.BOTH,
            expand=True
        )

        for column in range(3):
            self.thumb_frame.columnconfigure(
                column,
                weight=1
            )

        self.load_page(0)

    def confirm_multi_selection(self):

        if len(self.selected_reference_ids) < 2:
            return

        reference_ids = list(self.selected_reference_ids)
        self.displayed_reference_ids = reference_ids

        print(
            "MULTI START:",
            reference_ids,
            flush=True
        )

        similarity_maps = []

        for reference_id in reference_ids:

            print(
                f"Starting reference {reference_id}",
                flush=True
            )

            similarity_map = calculate_similarities_for_reference(
                self.repo,
                reference_id,
                self.embedding_index,
                candidate_count=MULTIPLE_CANDIDATE_COUNT,
            )

            print(
                f"Finished reference {reference_id}: "
                f"{len(similarity_map)} candidates",
                flush=True
            )

            similarity_maps.append(similarity_map)

        print(
            "All similarity searches finished",
            flush=True
        )

        candidate_ids = set()
        for similarity_map in similarity_maps:
            candidate_ids.update(similarity_map.keys())

        results = []

        for image_id in candidate_ids:

            # Die ausgewählten Referenzbilder selbst nicht als Treffer anzeigen.
            if image_id in self.selected_reference_ids:
                continue

            color_scores = []
            embedding_scores = []
            hash_scores = []

            for similarity_map in similarity_maps:
                scores = similarity_map.get(image_id)

                if scores is None:
                    continue

                color_scores.append(
                    scores.get("color", 0.0)
                )
                embedding_scores.append(
                    scores.get("embedding", 0.0)
                )
                hash_scores.append(
                    scores.get("hash", 0.0)
                )

            if not color_scores:
                continue

            combined_color = sum(color_scores) / len(color_scores)
            combined_embedding = (
                sum(embedding_scores) / len(embedding_scores)
            )
            combined_hash = sum(hash_scores) / len(hash_scores)

            combined_overall = (
                combined_color * self.color_weight.get()
                + combined_embedding * self.embedding_weight.get()
                + combined_hash * self.hash_weight.get()
            )

            results.append(
                (
                    combined_overall,
                    image_id,
                    combined_color,
                    combined_embedding,
                    combined_hash,
                )
            )

        results.sort(
            key=lambda row: row[0],
            reverse=True
        )

        self.current_results = results
        self.sort_column = "overall"
        self.sort_descending = True

        print(
            "Multi-reference candidates:",
            len(results),
            flush=True
        )

        self._show_top5(
            self.current_results,
            self.sort_column,
        )

        for rank, (
            overall,
            image_id,
            color,
            embedding,
            hash_value
        ) in enumerate(results[:5], start=1):

            image = self.repo.get_image_by_id(image_id)
            if image is None:
                continue

            print(
                f"#{rank}",
                image["filename"],
                f"overall={overall:.4f}",
                f"color={color:.4f}",
                f"embedding={embedding:.4f}",
                f"hash={hash_value:.4f}",
                flush=True
            )

        self.selected_reference_ids.clear()
        self._update_multi_selection_ui()

    def _update_multi_selection_ui(self):

        count = len(self.selected_reference_ids)

        if count >= 1:
            self.multi_selection_frame.pack(
                fill=tk.X,
                pady=(5, 0)
            )

            self.multi_selection_label.config(
                text=f"{count} Bild"
                + (" ausgewählt" if count == 1 else "er ausgewählt")
            )

            self.confirm_selection_button.config(
                state=(
                    tk.NORMAL
                    if count >= 2
                    else tk.DISABLED
                )
            )

        else:
            self.multi_selection_frame.pack_forget()

    def _go_to_entered_page(self, event=None):
        try:
            page = int(self.page_entry.get())
        except ValueError:
            return

        page -= 1

        if page < 0:
            page = 0

        if page >= self.total_pages:
            page = self.total_pages - 1

        self.load_page(page)

    def change_page(self, amount):
        new_page = self.current_page + amount

        if new_page < 0:
            new_page = 0

        if new_page >= self.total_pages:
            new_page = self.total_pages - 1

        if new_page == self.current_page:
            return

        self.load_page(new_page)

    def load_page(self, page):

        if page < 0 or page >= self.total_pages:
            return

        self.current_page = page

        self.images = self.repo.get_images_for_page(
            page,
            self.page_size
        )

        self.image_by_id = {
            row["id"]: row
            for row in self.images
        }

        self._clear_thumbnail_page()
        self._build_thumbnails()

        self.page_entry.delete(0, tk.END)
        self.page_entry.insert(
            0,
            str(self.current_page + 1)
        )

        self.page_total_label.config(
            text=f"von {self.total_pages}"
        )

        self.previous_button.config(
            state=(
                tk.NORMAL
                if self.current_page > 0
                else tk.DISABLED
            )
        )

        self.next_button.config(
            state=(
                tk.NORMAL
                if self.current_page < self.total_pages - 1
                else tk.DISABLED
            )
        )
        
        self.previous_1000_button.config(
            state=(
                tk.NORMAL
                if self.current_page >= 1000
                else tk.DISABLED
            )
        )

        self.previous_100_button.config(
            state=(
                tk.NORMAL
                if self.current_page >= 100
                else tk.DISABLED
            )
        )

        self.previous_10_button.config(
            state=(
                tk.NORMAL
                if self.current_page >= 10
                else tk.DISABLED
            )
        )

        self.previous_button.config(
            state=(
                tk.NORMAL
                if self.current_page >= 1
                else tk.DISABLED
            )
        )

        self.next_button.config(
            state=(
                tk.NORMAL
                if self.current_page + 1 < self.total_pages
                else tk.DISABLED
            )
        )

        self.next_10_button.config(
            state=(
                tk.NORMAL
                if self.current_page + 10 < self.total_pages
                else tk.DISABLED
            )
        )

        self.next_100_button.config(
            state=(
                tk.NORMAL
                if self.current_page + 100 < self.total_pages
                else tk.DISABLED
            )
        )

        self.next_1000_button.config(
            state=(
                tk.NORMAL
                if self.current_page + 1000 < self.total_pages
                else tk.DISABLED
            )
        )

    def previous_page(self):

        self.load_page(
            self.current_page - 1
        )


    def next_page(self):

        self.load_page(
            self.current_page + 1
        )


    def _clear_thumbnail_page(self):

        for widget in self.thumb_frame.winfo_children():
            widget.destroy()

        self.photo_refs.clear()

    def _build_thumbnails(self):

        columns = 3

        for idx, row in enumerate(self.images):

            frame = ttk.Frame(
                self.thumb_frame,
                padding=5,
                relief="ridge"
            )

            frame.grid(
                row=idx // columns,
                column=idx % columns,
                sticky="nsew",
                padx=3,
                pady=3
            )

            try:

                img = load_display_image(
                    row["filepath"]
                )

                img.thumbnail(THUMB_SIZE)

                thumb = Image.new(
                    "RGB",
                    THUMB_SIZE,
                    "white"
                )

                x = (
                    THUMB_SIZE[0] - img.width
                ) // 2

                y = (
                    THUMB_SIZE[1] - img.height
                ) // 2

                thumb.paste(
                    img,
                    (x, y)
                )

                photo = ImageTk.PhotoImage(
                    thumb
                )

                # Referenzen erhalten
                frame.photo = photo

                image_label = tk.Label(
                    frame,
                    image=photo,
                    cursor="hand2"
                )

                image_label.pack()

                image_label.bind(
                    "<Button-1>",
                    lambda e, image_id=row["id"]:
                        self.handle_thumbnail_click(e, image_id)
                )

            except Exception as e:

                ttk.Label(
                    frame,
                    text="Bild nicht verfügbar"
                ).pack()

                print(
                    f"Thumbnail error: "
                    f"{row['filepath']}: {e}"
                )

            name = ttk.Label(
                frame,
                text=row["filename"],
                wraplength=150,
                justify="center",
                cursor="hand2"
            )

            name.pack(
                fill=tk.X,
                pady=(4, 0)
            )

            name.bind(
                "<Button-1>",
                lambda e, image_id=row["id"]:
                    self.handle_thumbnail_click(e, image_id)
            )

    def _clear_top5(self):
        for widget in self.top5_frame.winfo_children():
            widget.destroy()
        self.top5_refs.clear()

    def _show_top5(self, results, sort_column="overall"):
        self._clear_top5()

        # -------------------------------------------------
        # Referenzbilder bestimmen
        # -------------------------------------------------
        reference_ids = []

        if self.displayed_reference_ids:
            reference_ids = list(self.displayed_reference_ids)
        elif self.selected_id is not None:
            reference_ids = [self.selected_id]

        # -------------------------------------------------
        # Ergebnisse sortieren
        # -------------------------------------------------
        index = {
            "overall": 0,
            "image": 1,
            "color": 2,
            "embedding": 3,
            "hash": 4,
        }

        if sort_column == "image":
            def image_name(result):
                image = self.repo.get_image_by_id(result[1])
                return image["filename"].lower() if image else ""

            ranked_results = sorted(
                results,
                key=image_name,
                reverse=self.sort_descending
            )
        else:
            ranked_results = sorted(
                results,
                key=lambda row: row[index[sort_column]],
                reverse=self.sort_descending
            )

        method_names = {
            "overall": "Gewichtet",
            "image": "Bild",
            "color": "Farbe",
            "embedding": "Embedding",
            "hash": "Hash",
        }

        # -------------------------------------------------
        # Layout
        # -------------------------------------------------
        # 0 = Referenzen, 1 = Referenz-Scrollbar,
        # 2/3 = Top-5-Ergebnisse
        for column in range(4):
            self.top5_frame.columnconfigure(
                column,
                weight=0
            )

        self.top5_frame.columnconfigure(2, weight=1)
        self.top5_frame.columnconfigure(3, weight=1)

        ttk.Label(
            self.top5_frame,
            text=f"Top 5 nach {method_names[sort_column]}",
            font=("TkDefaultFont", 11, "bold")
        ).grid(
            row=0,
            column=2,
            columnspan=2,
            sticky="w",
            pady=(0, 5)
        )

        # -------------------------------------------------
        # Referenzbereich
        # -------------------------------------------------
        if reference_ids:
            reference_canvas = tk.Canvas(
                self.top5_frame,
                width=220,
                height=450,
                highlightthickness=0
            )

            reference_scrollbar = ttk.Scrollbar(
                self.top5_frame,
                orient="vertical",
                command=reference_canvas.yview
            )

            reference_container = ttk.Frame(reference_canvas)

            reference_canvas.create_window(
                (0, 0),
                window=reference_container,
                anchor="nw"
            )

            reference_canvas.configure(
                yscrollcommand=reference_scrollbar.set
            )

            reference_container.bind(
                "<Configure>",
                lambda e: reference_canvas.configure(
                    scrollregion=reference_canvas.bbox("all")
                )
            )

            def scroll_references(event):
                reference_canvas.yview_scroll(
                    -1 if event.delta > 0 else 1,
                    "units"
                )
                return "break"

            reference_canvas.bind(
                "<MouseWheel>",
                scroll_references
            )

            reference_canvas.grid(
                row=1,
                column=0,
                rowspan=3,
                sticky="nsew",
                padx=(5, 0),
                pady=3
            )

            reference_scrollbar.grid(
                row=1,
                column=1,
                rowspan=3,
                sticky="ns",
                padx=(0, 5),
                pady=3
            )

            for reference_index, reference_id in enumerate(
                reference_ids,
                start=1
            ):
                reference = self.repo.get_image_by_id(reference_id)

                if reference is None:
                    continue

                ref_card = ttk.Frame(
                    reference_container,
                    padding=6,
                    relief="ridge"
                )
                ref_card.pack(
                    fill=tk.X,
                    padx=3,
                    pady=3
                )

                try:
                    img = load_display_image(reference["filepath"])
                    img.thumbnail(TOP_RESULT_SIZE)

                    ref_image = Image.new(
                        "RGB",
                        TOP_RESULT_SIZE,
                        "white"
                    )

                    x = (TOP_RESULT_SIZE[0] - img.width) // 2
                    y = (TOP_RESULT_SIZE[1] - img.height) // 2
                    ref_image.paste(img, (x, y))

                    photo = ImageTk.PhotoImage(ref_image)
                    self.top5_refs.append(photo)

                    tk.Label(
                        ref_card,
                        image=photo
                    ).pack()

                except Exception as exc:
                    ttk.Label(
                        ref_card,
                        text=f"Bild nicht verfügbar\n{exc}",
                        width=28,
                        anchor="center"
                    ).pack(pady=40)

                label_text = (
                    f"REFERENZ {reference_index}"
                    if len(reference_ids) >= 2
                    else "REFERENZ"
                )

                ttk.Label(
                    ref_card,
                    text=label_text,
                    font=("TkDefaultFont", 11, "bold")
                ).pack(pady=(5, 2))

                ttk.Label(
                    ref_card,
                    text=reference["filename"],
                    wraplength=210,
                    justify="center"
                ).pack(fill=tk.X)

        # -------------------------------------------------
        # Top 5
        # -------------------------------------------------
        positions = [
            (1, 2),
            (1, 3),
            (2, 2),
            (2, 3),
            (3, 2),
        ]

        for rank, (
            overall,
            image_id,
            color,
            embedding,
            hash_value
        ) in enumerate(ranked_results[:5], start=1):

            row = self.repo.get_image_by_id(image_id)
            if row is None:
                continue

            result_row, result_column = positions[rank - 1]

            card = ttk.Frame(
                self.top5_frame,
                padding=6,
                relief="ridge"
            )

            card.grid(
                row=result_row,
                column=result_column,
                padx=5,
                pady=3,
                sticky="nsew"
            )

            try:
                img = load_display_image(row["filepath"])
                img.thumbnail(TOP_RESULT_SIZE)

                card_image = Image.new(
                    "RGB",
                    TOP_RESULT_SIZE,
                    "white"
                )

                x = (TOP_RESULT_SIZE[0] - img.width) // 2
                y = (TOP_RESULT_SIZE[1] - img.height) // 2
                card_image.paste(img, (x, y))

                photo = ImageTk.PhotoImage(card_image)
                self.top5_refs.append(photo)

                image_label = tk.Label(
                    card,
                    image=photo,
                    cursor="hand2"
                )
                image_label.pack()

                image_label.bind(
                    "<Button-1>",
                    lambda e, image_id=image_id:
                    self.select_reference(image_id)
                )

            except Exception as exc:
                ttk.Label(
                    card,
                    text=f"Bild nicht verfügbar\n{exc}",
                    width=28,
                    anchor="center"
                ).pack(pady=40)

            filename = row["filename"]

            max_length = 24

            if len(filename) > max_length:
                filename = filename[:max_length - 3] + "..."

            ttk.Label(
                card,
                text=f"#{rank}  {filename}",
                justify="center"
            ).pack(
                fill=tk.X,
                pady=(5, 2)
            )

            if sort_column == "overall":
                selected_score = overall
            elif sort_column == "color":
                selected_score = color
            elif sort_column == "embedding":
                selected_score = embedding
            elif sort_column == "hash":
                selected_score = hash_value
            else:
                selected_score = None

            if selected_score is not None:
                ttk.Label(
                    card,
                    text=(
                        f"{method_names[sort_column]}: "
                        f"{selected_score:.4f}"
                    ),
                    font=("TkDefaultFont", 10, "bold")
                ).pack()

            ttk.Label(
                card,
                text=(
                    f"Farbe {color:.4f}\n"
                    f"Embedding {embedding:.4f}\n"
                    f"Hash {hash_value:.4f}"
                ),
                justify="center"
            ).pack()

    def handle_thumbnail_click(self, event, image_id):
        # Shift gedrückt?
        if event.state & 0x0001:
            if image_id in self.selected_reference_ids:
                self.selected_reference_ids.remove(image_id)
            else:
                self.selected_reference_ids.add(image_id)

            self._update_multi_selection_ui()
            return

        # Normaler Klick
        self.selected_reference_ids.clear()
        self.select_reference(image_id)
        self._update_multi_selection_ui()

    def select_reference(self, reference_id):
        self.selected_reference_ids.clear()
        self.displayed_reference_ids = []
        self.selected_id = reference_id

        ref = self.repo.get_image_by_id(reference_id)

        if ref is None:
            return

        self.image_by_id[reference_id] = ref

        self.reference_label.config(
            text=f"Referenz: {ref['filename']}"
        )

        self.reference_score.config(
            text=f"{ref['width']} × {ref['height']} px  |  {ref['filepath']}"
        )

        similarity_map = calculate_similarities_for_reference(
            self.repo,
            reference_id,
            self.embedding_index,
            candidate_count=CANDIDATE_COUNT,
        )

        results = []

        for image_id, scores in similarity_map.items():

            image = self.repo.get_image_by_id(image_id)

            if image is None:
                continue

            self.image_by_id[image_id] = image

            color = scores.get("color", 0.0)
            embedding = scores.get("embedding", 0.0)
            hash_value = scores.get("hash", 0.0)

            overall = (
                color * self.color_weight.get()
                + embedding * self.embedding_weight.get()
                + hash_value * self.hash_weight.get()
            )

            results.append(
                (
                    overall,
                    image_id,
                    color,
                    embedding,
                    hash_value,
                )
            )

        self.current_results = results

        self.sort_column = "overall"
        self.sort_descending = True

        self.current_results.sort(
            key=lambda row: row[0],
            reverse=True,
        )

        self._show_top5(
            self.current_results,
            self.sort_column,
        )

        self.selected_reference_ids.clear()
        self._update_multi_selection_ui()
    
    def _build_weight_controls(self, parent):

        frame = ttk.LabelFrame(
            parent,
            text="Similarity-Gewichtung",
            padding=10
        )

        frame.pack(
            fill=tk.X,
            pady=(0, 10)
        )

        self.color_weight_label = ttk.Label(frame)
        self.color_weight_label.grid(
            row=0,
            column=0,
            sticky="w"
        )

        color_scale = ttk.Scale(
            frame,
            from_=0,
            to=1,
            variable=self.color_weight,
            orient=tk.HORIZONTAL,
            command=self._update_weight_labels
        )

        color_scale.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=10
        )
        
        color_scale.bind(
            "<Button-1>",
            lambda event: self._scale_click(
                event,
                color_scale,
                self.color_weight
            )
        )

        self.embedding_weight_label = ttk.Label(frame)
        self.embedding_weight_label.grid(
            row=1,
            column=0,
            sticky="w"
        )

        embedding_scale = ttk.Scale(
            frame,
            from_=0,
            to=1,
            variable=self.embedding_weight,
            orient=tk.HORIZONTAL,
            command=self._update_weight_labels
        )

        embedding_scale.grid(
            row=1,
            column=1,
            sticky="ew",
            padx=10
        )
        
        embedding_scale.bind(
            "<Button-1>",
            lambda event: self._scale_click(
                event,
                embedding_scale,
                self.embedding_weight
            )
        )

        self.hash_weight_label = ttk.Label(frame)
        self.hash_weight_label.grid(
            row=2,
            column=0,
            sticky="w"
        )

        hash_scale = ttk.Scale(
            frame,
            from_=0,
            to=1,
            variable=self.hash_weight,
            orient=tk.HORIZONTAL,
            command=self._update_weight_labels
        )

        hash_scale.grid(
            row=2,
            column=1,
            sticky="ew",
            padx=10
        )
        
        hash_scale.bind(
            "<Button-1>",
            lambda event: self._scale_click(
                event,
                hash_scale,
                self.hash_weight
            )
        )

        frame.columnconfigure(1, weight=1)

        ttk.Button(
            frame,
            text="Gewichtung anwenden",
            command=self._apply_weights
        ).grid(
            row=3,
            column=0,
            columnspan=2,
            pady=(8, 0)
        )

        self._update_weight_labels()
    
    def _update_weight_labels(self, *args):

        self.color_weight_label.config(
            text=f"Farbe: {self.color_weight.get() * 100:.0f}%"
        )

        self.embedding_weight_label.config(
            text=f"Embedding: {self.embedding_weight.get() * 100:.0f}%"
        )

        self.hash_weight_label.config(
            text=f"Hash: {self.hash_weight.get() * 100:.0f}%"
        )
    
    def _apply_weights(self):

        color = self.color_weight.get()
        embedding = self.embedding_weight.get()
        hash_value = self.hash_weight.get()

        total = color + embedding + hash_value

        if total == 0:
            return

        color /= total
        embedding /= total
        hash_value /= total

        self.color_weight.set(color)
        self.embedding_weight.set(embedding)
        self.hash_weight.set(hash_value)

        self._update_weight_labels()

        if not self.current_results:
            return

        new_results = []

        for (
            _overall,
            image_id,
            color_score,
            embedding_score,
            hash_score
        ) in self.current_results:

            overall = (
                color_score * color
                + embedding_score * embedding
                + hash_score * hash_value
            )

            new_results.append(
                (
                    overall,
                    image_id,
                    color_score,
                    embedding_score,
                    hash_score,
                )
            )

        self.current_results = new_results

        self.current_results.sort(
            key=lambda row: row[0],
            reverse=self.sort_descending
        )

        self._show_top5(
            self.current_results,
            self.sort_column
        )

    def _scale_click(self, event, scale, variable):

        width = scale.winfo_width()

        if width <= 1:
            return

        x = max(0, min(event.x, width))
        value = x / width

        scale.after(
            1,
            lambda: (
                variable.set(value),
                self._update_weight_labels()
            )
        )