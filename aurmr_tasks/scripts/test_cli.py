#!/usr/bin/env python

from typing import List
from typing import Tuple

from rich.console import Console

from rich.progress import track
import rich_click as click
from click_prompt import choice_option
import questionary

import rospy
from aurmr_tasks.srv import StowRequest, PickRequest

def get_items() -> List[Tuple[str, str]]:
    return [
        ("4E", "mommys_bliss_melatonin", "B079TD7HG2"),
            ("4E", "testing", ""),
            ("4F", "testing", ""),
            ("3E", "testing", ""),
            ("3F", "vitamins", ""),
            ("3F", "lysol", ""),
            ("2H", "airpods", ""),
            ("2F", "mouthwash", ""),
            ("2F", "soup", ""),
            ("2F", "masks", ""),
            ("1F", "box", "")]

@click.command()
@choice_option('--items', type=click.Choice([f'{b}: {o} ({a})' for b,o,a in get_items()]), multiple=True)
def cli(items):

    console = Console()

    pick = rospy.ServiceProxy('/aurmr_demo/pick', PickRequest)
    stow = rospy.ServiceProxy('/aurmr_demo/stow', StowRequest)


    stowed_items = []

    console.rule('Stowing')

    print(items)
    for i in items:
        b, o_a = i.split(': ', maxsplit=1)
        o, a = o_a.split(" (")
        a = a[:-1]
        if questionary.confirm(f'Please confirm that {o} is stowed in bin {b}.').ask():
            stow(bin_id=b, object_id=o)
            stowed_items.append((b, o, a))
        else:
            console.print(f'Skipping item [orange]{o}[/orange]')

    console.rule('Picking')
    for b,o,a in track(stowed_items, console=console):
        console.print(f'Picking item [orange]{o}[/orange] out of bin [teal]{b}[teal]')
        pick(bin_id=b, object_id=o, object_asin=a)


if __name__ == "__main__":
    cli()
