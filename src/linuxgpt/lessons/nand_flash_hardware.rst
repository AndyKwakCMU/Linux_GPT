.. LESSON -- original synthesis, not verbatim kernel source.
.. Status: UNREVIEWED, pending Jaeguek Kim's review (see lessons/manifest.yaml).
.. Every fenced code/rst block below is a byte-verified quote of real
.. indexed kernel documentation; the connective narrative between them
.. is this lesson author's own synthesis and has not yet been checked
.. by a human F2FS/flash-storage expert.

================================================================
NAND Flash Hardware Fundamentals, and Why F2FS Is Shaped By It
================================================================

F2FS's design choices only make sense in light of the storage hardware
it targets. This lesson covers the NAND flash constraints that matter
most, then ties each one directly to a specific F2FS design decision
already documented in the indexed F2FS overview.

What NAND Flash Actually Is
============================

NAND flash is the storage technology behind SSDs, eMMC, and SD cards --
the devices F2FS was written for. The kernel's generic NAND driver
subsystem describes the class of hardware directly:

```rst file=Documentation/driver-api/mtdnand.rst lines=7-15
Introduction
============

The generic NAND driver supports almost all NAND and AG-AND based chips
and connects them to the Memory Technology Devices (MTD) subsystem of
the Linux Kernel.

This documentation is provided for developers who want to implement
board drivers or filesystem drivers suitable for NAND devices.
```

Two physical properties of this hardware drive almost everything else in
this lesson: writes happen at a small granularity (a "page", commonly a
few KB), but a page cannot be rewritten in place -- the much larger
"erase block" containing it (commonly hundreds of pages) must be erased
as a whole before any of its pages can be written again. This
erase-before-write asymmetry (fast, cheap writes to a fresh page; slow,
coarse-grained erases) is not covered in as much line-by-line detail in
the currently-indexed driver docs as it deserves -- treat this specific
framing as this lesson's own synthesis pending review, not a direct
quote -- but it is the physical fact underneath every design decision
below.

Reliability: Why Flash Needs Error Correction
================================================

Flash storage is also less reliable at the bit level than it might
appear -- individual bits can flip, especially as a device wears. This
is why NAND flash controllers universally rely on error-correcting
codes:

```rst file=Documentation/driver-api/mtd/nand_ecc.rst lines=19-24
The problem
===========

NAND flash (at least SLC one) typically has sectors of 256 bytes.
However NAND flash is not extremely reliable so some error detection
(and sometimes correction) is needed.
```

A filesystem or driver layer does not usually reimplement this itself
(it is handled by the flash controller / MTD ECC engine), but it is part
of why flash storage behaves differently from the "just a linear array
of reliable bytes" mental model that older filesystem designs assumed.

Why This Forces a Log-Structured Design
==========================================

Because individual flash pages cannot be updated in place without first
erasing their entire containing block, a filesystem that tried to
update data in place the way a traditional disk filesystem does would
force a full erase-block erase (and a copy of everything else still
live in that block) on nearly every write. F2FS avoids this entirely by
never overwriting in place -- it always appends to a fresh log, which is
precisely the Log-structured File System (LFS) approach:

```rst file=Documentation/filesystems/f2fs.rst lines=39-49
Log-structured File System (LFS)
--------------------------------
"A log-structured file system writes all modifications to disk sequentially in
a log-like structure, thereby speeding up  both file writing and crash recovery.
The log is the only structure on disk; it contains indexing information so that
files can be read back from the log efficiently. In order to maintain large free
areas on disk for fast writing, we divide  the log into segments and use a
segment cleaner to compress the live information from heavily fragmented
segments." from Rosenblum, M. and Ousterhout, J. K., 1992, "The design and
implementation of a log-structured file system", ACM Trans. Computer Systems
10, 1, 26–52.
```

This is a direct, causal link: erase-before-write (this lesson, above)
is why F2FS chose LFS (quoted directly from F2FS's own design
documentation) rather than an in-place-update design like a traditional
disk filesystem.

The Costs LFS Introduces -- and F2FS's Answers
=================================================

Choosing LFS to avoid the erase-before-write problem does not come for
free; F2FS's own documentation identifies two costs it specifically
had to design around. First, the "wandering tree" problem, where a
single leaf update can cascade into rewriting every index structure
above it:

```rst file=Documentation/filesystems/f2fs.rst lines=51-61
Wandering Tree Problem
----------------------
In LFS, when a file data is updated and written to the end of log, its direct
pointer block is updated due to the changed location. Then the indirect pointer
block is also updated due to the direct pointer block update. In this manner,
the upper index structures such as inode, inode map, and checkpoint block are
also updated recursively. This problem is called as wandering tree problem [1],
and in order to enhance the performance, it should eliminate or relax the update
propagation as much as possible.

[1] Bityutskiy, A. 2005. JFFS3 design issues. http://www.linux-mtd.infradead.org/
```

Second, cleaning overhead -- since old versions of data are never
overwritten in place, they become obsolete blocks scattered across the
device that eventually need to be reclaimed:

```rst file=Documentation/filesystems/f2fs.rst lines=63-80
Cleaning Overhead
-----------------
Since LFS is based on out-of-place writes, it produces so many obsolete blocks
scattered across the whole storage. In order to serve new empty log space, it
needs to reclaim these obsolete blocks seamlessly to users. This job is called
as a cleaning process.

The process consists of three operations as follows.

1. A victim segment is selected through referencing segment usage table.
2. It loads parent index structures of all the data in the victim identified by
   segment summary blocks.
3. It checks the cross-reference between the data and its parent index structure.
4. It moves valid data selectively.

This cleaning job may cause unexpected long delays, so the most important goal
is to hide the latencies to users. And also definitely, it should reduce the
amount of valid data to be moved, and move them quickly as well.
```

F2FS explicitly names flash awareness as a first-class design goal
alongside solving these two costs:

```rst file=Documentation/filesystems/f2fs.rst lines=85-89
Flash Awareness
---------------
- Enlarge the random write area for better performance, but provide the high
  spatial locality
- Align FS data structures to the operational units in FTL as best efforts
```

Putting It Together
======================

The causal chain, end to end: NAND flash physically cannot rewrite a
page without erasing its whole containing block (this lesson) -->
therefore a filesystem that avoids in-place updates entirely (LFS) is a
much better fit for flash than a traditional disk filesystem's
in-place-update design (this lesson, quoting F2FS's own rationale) -->
which in turn creates the wandering-tree and cleaning-overhead problems
that most of F2FS's segment/checkpoint machinery (see the F2FS-VFS-
block-mm lesson, and the checkpoint-related source chunks already
indexed from ``fs/f2fs/checkpoint.c``) exists specifically to manage.
