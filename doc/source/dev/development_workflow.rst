Development workflow
--------------------

.. _development-workflow:

.. Project authors:
.. _Erik Ingwersen: https://github.com/ingwersen-erik
.. _erikingwersen@gmail.com: erikingwersen@gmail.com
.. _Numpy: https://numpy.org/doc/stable/dev/development_workflow.html
.. _github: https://github.com/
.. _git push: https://www.atlassian.com/git/tutorials/syncing/git-push
.. _git_cheat_sheet: https://education.github.com/git-cheat-sheet-education.pdf

.. _LICENSE: <../../../LICENSE>

.. bibliographic fields:

:Author: `Erik Ingwersen`_
:Contact: `erikingwersen@gmail.com`_
:date: $Date: 2021-12-04 8:53:53 +0000 (Sat, 04 Dev 2021) $
:status: This is a "work in progress"
:version: 1.00
:copyright: `LICENSE`_


.. meta::
   :keywords: Pandas, inputs preparation, datatools, CI/CD, DevOps, GitHub, Workflow
   :description lang=en: Tutorial on how to maintain a good development workflow

:abstract:
    This page focuses on some good development practices, related to development
    workflow. What is described below is a recommended workflow with Git, in a 
    condensed format. For more information, visit `Numpy`_ development 
    documentation, as this section was copied from there.

-------

Basic workflow
#############

In short:

1. Start a new *feature branch* for each set of edits that you do.
   See `making-a-new-feature-branch`_ below.

2. Hack away! See `editing-workflow`_ below.

This way of working helps to keep work well organized and the history
as clear as possible.

.. seealso::

   There are many online tutorials to help you `learn git`_. For discussions
   of specific git workflows, see these discussions on `linux git workflow`_,
   and `ipython git workflow`_.
   
   See `git_cheat_sheet`_ for a quick refresher on which ``git`` commands from are supported on GitHub.

.. _making-a-new-feature-branch:

Making a new feature branch
===========================

First, fetch new commits from the ``upstream`` repository:

::

   git fetch upstream

Then, create a new branch based on the main branch of the upstream
repository::

   git checkout -b my-new-feature upstream/main


.. _editing-workflow:

The editing workflow
====================

Overview
--------

::

   # hack hack
   git status # Optional
   git diff # Optional
   git add modified_file
   git commit
   # push the branch to your own GitHub repo
   git push origin my-new-feature

In more detail
--------------

#. Make some changes. When you feel that you've made a complete, working set
   of related changes, move on to the next steps.

#. Optional: Check which files have changed with ``git status`` (see `git status`_).  You'll see a listing like this one::

     # On branch my-new-feature
     # Changed but not updated:
     #   (use "git add <file>..." to update what will be committed)
     #   (use "git checkout -- <file>..." to discard changes in working directory)
     #
     #	modified:   README
     #
     # Untracked files:
     #   (use "git add <file>..." to include in what will be committed)
     #
     #	INSTALL
     no changes added to commit (use "git add" and/or "git commit -a")

#. Optional: Compare the changes with the previous version using with (``git diff``). 
   This brings up a simple text browser interface that highlights the difference between 
   your files and the previous version.

#. Add any relevant modified or new files using  ``git add modified_file``
   (see ``git add``). This puts the files into a staging area, which is a queue
   of files that will be added to your next commit. Only add files that have
   related, complete changes. Leave files with unfinished changes for later
   commits.

#. To commit the staged files into the local copy of your repo, do ``git
   commit``. At this point, a text editor will open up to allow you to write a
   commit message. Read the `writing-the-commit-message` section to be sure that you are writing a
   properly formatted and sufficiently detailed commit message. After saving
   your message and closing the editor, your commit will be saved. For trivial
   commits, a short commit message can be passed in through the command line
   using the ``-m`` flag. For example, ``git commit -am "ENH: Some message"``.

   In some cases, you will see this form of the commit command: ``git commit
   -a``. The extra ``-a`` flag automatically commits all modified files and
   removes all deleted files. This can save you some typing of numerous ``git
   add`` commands; however, it can add unwanted changes to a commit if you're
   not careful.

#. Push the changes to your forked repo on `github`_::

      git push origin my-new-feature

   For more information, see `git push`_.

.. note::

   Assuming you have followed the instructions in these pages, git will create
   a default link to your `github`_ repo called ``origin``.  In git >= 1.7 you
   can ensure that the link to origin is permanently set by using the
   ``--set-upstream`` option::

      git push --set-upstream origin my-new-feature

   From now on ``git`` will know that ``my-new-feature`` is related to the
   ``my-new-feature`` branch in your own github_ repo. Subsequent push calls
   are then simplified to the following::

      git push

   You have to use ``--set-upstream`` for each new branch that you create.


It may be the case that while you were working on your edits, new commits have
been added to ``upstream`` that affect your work.

.. _writing-the-commit-message:

Writing the commit message
--------------------------

Commit messages should be clear and follow a few basic rules.  Example::

   ENH: add functionality X to datatools.<submodule>.

   The first line of the commit message starts with a capitalized acronym
   (options listed below) indicating what type of commit this is.  Then a blank
   line, then more text if needed.  Lines shouldn't be longer than 72
   characters.  If the commit is related to a ticket, indicate that with
   "See #3456", "See ticket 3456", "Closes #3456" or similar.

Describing the motivation for a change, the nature of a bug for bug fixes or
some details on what an enhancement does are also good to include in a commit
message.  Messages should be understandable without looking at the code
changes.  A commit message like ``MAINT: fixed another one`` is an example of
what not to do; the reader has to go look for context elsewhere.

Standard acronyms to start the commit message with are::

   API: an (incompatible) API change
   BENCH: changes to the benchmark suite
   BLD: change related to building
   BUG: bug fix
   DEP: deprecate something, or remove a deprecated object
   DEV: development tool or utility
   DOC: documentation
   ENH: enhancement
   MAINT: maintenance commit (refactoring, typos, etc.)
   REV: revert an earlier commit
   STY: style fix (whitespace, PEP8)
   TST: addition or modification of tests
   REL: related to releasing


**TO-DO:** Finish workflow tutorial!
