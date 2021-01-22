from rich.console import Console
from rich.table import Table
import csv

console = Console()
table = Table(show_header=True, header_style="bold magenta")

with open('note.csv') as f:
    f_csv = csv.reader(f)
    fields = next(f_csv)

    for x in fields:
        table.add_column(x, justify="left")

    for row in f_csv:
        table.add_row(*row)

console.print(table)