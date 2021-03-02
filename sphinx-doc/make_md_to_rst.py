
"""
The script generates a folder of rest files from a folder of markdown files.
Markdown Hyperlinks between the files in the folder get converted to rest links so that they function properly in a
sphinx generated html build obtained from the resulting rest folder.
"""

import os
import click
import shutil
from m2r import parse_from_file
import re

new_line_re = "(\r\n|[\r\n])"

def rebaseAbsRoot(path, target, root):
    """
    If path and target intersect at root, return relative path from path to target
    Functionality is limited.
    path and target must be path strings pointing at FILES!
    root is only allowed to appear once in every path
    you cant root to os.sep (no folder seperators allowed in the root string)
    """

    p = path.find(root)
    t = target.find(root)
    if (p == -1) or (t == -1) or ('..' in path):
        return target

    path = path[path.find(root):].split(os.sep)
    target = target[target.find(root):].split(os.sep)
    # remove common path chunks:
    while path[0] == target[0]:
        del path[0]
        del target[0]

    up_steps = (len(path) - 1)*f'..{os.sep}'
    down_steps = os.sep.join(target)
    new_path = os.path.join(up_steps, down_steps)
    return new_path

def fixTables(f_rst):
    body_re = f'((.+){new_line_re})*{new_line_re}((.+){new_line_re})*'
    tables = list(re.finditer(f'\.\. list-table::{new_line_re}' + body_re, f_rst))
    for t in tables:
        tab = t[0]
        def pic_repl(match):
            leading = match.groupdict()['list_level']
            pic_dir = match.groupdict()['pic_directive']
            pic_pad = re.match('^[ ]*', pic_dir).span()[1]
            pic_dir = re.sub(f'{" " * pic_pad}', " " * len(leading), pic_dir)
            pic_dir = leading + pic_dir[len(leading):]
            end_space = re.search(f'{new_line_re}[ ]*$', match[0])
            if end_space:
                pic_dir = re.sub(f'{new_line_re}[ ]*$', end_space[0], pic_dir)
            return pic_dir
        messy_re = f'(?P<list_level>.*){new_line_re}(?P<pic_directive>[ ]*.. image::[^*-]*)'
        # using while loop cause messed pic patterns overlap
        tab, repnum = re.subn(messy_re, pic_repl, tab, 1)
        while repnum:
            tab, repnum = re.subn(messy_re, pic_repl, tab, 1)

        bullets = tab.split('   *')[1:]
        items = [bullet.split('     -') for bullet in bullets]
        last_items = items[-1]
        item_num = len(items[0])
        last_item_num = len(last_items)
        if item_num > last_item_num:
            has_content = len([content for content in last_items if re.search('[^\s-]', content)]) > 0
            if has_content:
                # append empty cells
                tab = tab + ('     - \n'*(item_num - last_item_num))
            else:
                # delete last row (using replace to avoid false meta char interpretation
                tab = tab.replace(bullets[-1][0], '')

        bullet_num = len(list(re.finditer(f'   \*(?P<items>([ ]+-.*{new_line_re})*)', tab)))
        if bullet_num == 1:
            #fix empty body table error:
            tab = re.sub(':header-rows: [0-9]', ':header-rows: 0', tab)

        if tab != t[0]:
            f_rst = f_rst.replace(t[0], tab)

    return f_rst


def fixLinks(f_rst, f ,targetpath):
    md_links = list(
        re.finditer('(?P<numbered>\. )?`(?P<link_name>[^<`]*) <(?P<md_link>\S*.md)?(#)?(?P<section>[^>]*)?>`_?', f_rst))
    for link in md_links:
        # change directory:
        link_path = link.groupdict()['md_link']
        if not link_path:
            link_path = f
        # change directory to point at temporal rest dir (if link isnt relative):
        if os.path.dirname(link_path) is not '':
            link_path = os.path.join(os.path.dirname(link_path) + '_m2r', os.path.basename(link_path))
        # rebase the link to relative link if its not
        link_path = rebaseAbsRoot(os.path.join(targetpath, f), link_path, 'sphinx-doc')
        # remove extension name (rst syntax)
        link_path = re.sub('\.md$', '', link_path)
        if link.groupdict()['section']:
            # while document links have to be relative - section links have to be absolute from sphinx doc dir -
            # markdown space representation by dash has to be removed...
            abs_path = os.path.basename(os.path.abspath(''))
            abs_path = targetpath[targetpath.find(abs_path) + len(abs_path) + 1:]
            link_path = os.path.join(abs_path, os.path.basename(link_path))
            role = ':ref:'
            section = ':' + link.groupdict()['section'].replace('-', ' ')
            # one more regex spell for the sake of numbered section linking:
            if link.groupdict()['numbered']:
                section = re.sub('(:[0-9]+)', '\g<1>.', section)
        else:
            role = ':doc:'
            section = ''

        f_rst = re.sub(f'`(?P<link_name>{link.groupdict()["link_name"]}) '
                       f'<({link.groupdict()["md_link"]})?(#[^>]*)?>`(_)?',
                       r'{}`\g<link_name> <{}{}>`'.format(role, link_path, section), f_rst)
    return f_rst


@click.command()
@click.option(
    "-p", "--mdpath", type=str,  required=True, default="sphinx-doc/getting_started_md",
    help="Relative path to the folder containing the .md files to be converted (relative to sphinx root)."
)
@click.option(
    "-sr", "--sphinxroot", type=str,  required=True, default='..', help="Relative path to the sphinx root."
)
def main(mdpath, sphinxroot):
    root_path = os.path.abspath(sphinxroot)
    mdpath = os.path.join(root_path, mdpath)
    targetpath = mdpath + "_m2r"

    # clear target directory:
    if os.path.isdir(targetpath):
        shutil.rmtree(targetpath)
    os.mkdir(targetpath)

    mdfiles = [f for f in os.listdir(mdpath) if os.path.splitext(f)[1] == '.md']
    for f in mdfiles:
        f_rst = parse_from_file(os.path.join(mdpath, f))
        # regex magic- replace invalid links:
        f_rst = fixLinks(f_rst, f, targetpath)
        f_rst = fixTables(f_rst)
        with open(os.path.join(targetpath, f.replace('.md', '.rst')), 'w+') as file_:
            file_.write(f_rst)


if __name__ == "__main__":
    main()