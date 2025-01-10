from datasets import load_dataset

book_sum_sort = load_dataset("tau/zero_scrolls", "book_sum_sort", split="test")
"""
Options are: ["gov_report", "summ_screen_fd", "qmsum", "squality", "qasper","narrative_qa", "quality", "musique", "space_digest","book_sum_sort"]
There is also a small number of examples (~20 per task) in a "validation" split, meant for eyeballing purposes
"""
book_sum_sort.save_to_disk("./book_sum_sort")