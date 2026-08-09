.. LESSON -- original synthesis, not verbatim kernel source.
.. Status: UNREVIEWED, pending Jaeguek Kim's review (see lessons/manifest.yaml).
.. Every fenced code/rst block below is a byte-verified quote of real
.. indexed kernel source or documentation; the connective narrative
.. between them is this lesson author's own synthesis and has not yet
.. been checked by a human F2FS expert.

===================================================================
F2FS, the VFS, the Block Layer, and the Page Cache: How They Fit
===================================================================

F2FS is not a standalone piece of software -- it is one filesystem
implementation plugged into three layers the kernel already provides:
the Virtual File System (VFS), the block I/O layer, and the page cache.
Understanding F2FS's own log-structured design (see the F2FS overview
lesson content already indexed from ``Documentation/filesystems/f2fs.rst``)
only tells part of the story; this lesson traces the boundary between
F2FS and each of those three layers, using F2FS's actual registration
structs and I/O entry points as evidence.

F2FS as a VFS Client
====================

The VFS is the kernel's abstraction layer that lets many different
filesystem implementations present the same system-call interface
(``open``, ``read``, ``write``, ``stat``, ...) to userspace:

```rst file=Documentation/filesystems/vfs.rst lines=13-24
Introduction
============

The Virtual File System (also known as the Virtual Filesystem Switch) is
the software layer in the kernel that provides the filesystem interface
to userspace programs.  It also provides an abstraction within the
kernel which allows different filesystem implementations to coexist.

VFS system calls open(2), stat(2), read(2), write(2), chmod(2) and so on
are called from a process context.  Filesystem locking is described in
the document Documentation/filesystems/locking.rst.

```

A filesystem joins the VFS by registering a ``file_system_type`` (so the
kernel knows how to mount it) and, once mounted, a ``super_operations``
table (so the VFS knows how to manage that mounted instance -- allocating
inodes, syncing, unmounting, and so on). F2FS's registration is a direct,
literal instance of this:

```c file=fs/f2fs/super.c lines=4936-4942
static struct file_system_type f2fs_fs_type = {
	.owner		= THIS_MODULE,
	.name		= "f2fs",
	.mount		= f2fs_mount,
	.kill_sb	= kill_f2fs_super,
	.fs_flags	= FS_REQUIRES_DEV | FS_ALLOW_IDMAP,
};
```

And its superblock operations table -- what the VFS calls to allocate/free
F2FS's own inode representation, write a dirty inode back, freeze/unfreeze
the filesystem for snapshotting, report statfs(2) info, and more:

```c file=fs/f2fs/super.c lines=3151-3171
static const struct super_operations f2fs_sops = {
	.alloc_inode	= f2fs_alloc_inode,
	.free_inode	= f2fs_free_inode,
	.drop_inode	= f2fs_drop_inode,
	.write_inode	= f2fs_write_inode,
	.dirty_inode	= f2fs_dirty_inode,
	.show_options	= f2fs_show_options,
#ifdef CONFIG_QUOTA
	.quota_read	= f2fs_quota_read,
	.quota_write	= f2fs_quota_write,
	.get_dquots	= f2fs_get_dquots,
#endif
	.evict_inode	= f2fs_evict_inode,
	.put_super	= f2fs_put_super,
	.sync_fs	= f2fs_sync_fs,
	.freeze_fs	= f2fs_freeze,
	.unfreeze_fs	= f2fs_unfreeze,
	.statfs		= f2fs_statfs,
	.remount_fs	= f2fs_remount,
	.shutdown	= f2fs_shutdown,
};
```

The VFS's own description of what a superblock object represents is
intentionally generic, precisely because it has to describe every
filesystem, not just F2FS:

```rst file=Documentation/filesystems/vfs.rst lines=238-242
The Superblock Object
=====================

A superblock object represents a mounted filesystem.

```

Inode and File Operations: Where F2FS's Behavior Actually Lives
=================================================================

``super_operations`` covers the filesystem as a whole. The VFS also
defines per-inode and per-open-file dispatch tables -- ``inode_operations``
(create, lookup, link, rename, permission checks, ...) and
``file_operations`` (read, write, mmap, fsync, ioctl, ...) -- and this is
where most of F2FS's actual filesystem semantics are implemented, as
plain C function pointers the VFS calls into. A regular file's inode
operations:

```c file=fs/f2fs/file.c lines=1103-1112
const struct inode_operations f2fs_file_inode_operations = {
	.getattr	= f2fs_getattr,
	.setattr	= f2fs_setattr,
	.get_inode_acl	= f2fs_get_acl,
	.set_acl	= f2fs_set_acl,
	.listxattr	= f2fs_listxattr,
	.fiemap		= f2fs_fiemap,
	.fileattr_get	= f2fs_fileattr_get,
	.fileattr_set	= f2fs_fileattr_set,
};
```

A directory's inode operations -- notably ``.lookup``/``.create``/``.unlink``,
the operations that make F2FS's on-disk directory format visible as
ordinary directory entries to the rest of the kernel:

```c file=fs/f2fs/namei.c lines=1316-1335
const struct inode_operations f2fs_dir_inode_operations = {
	.create		= f2fs_create,
	.lookup		= f2fs_lookup,
	.link		= f2fs_link,
	.unlink		= f2fs_unlink,
	.symlink	= f2fs_symlink,
	.mkdir		= f2fs_mkdir,
	.rmdir		= f2fs_rmdir,
	.mknod		= f2fs_mknod,
	.rename		= f2fs_rename2,
	.tmpfile	= f2fs_tmpfile,
	.getattr	= f2fs_getattr,
	.setattr	= f2fs_setattr,
	.get_inode_acl	= f2fs_get_acl,
	.set_acl	= f2fs_set_acl,
	.listxattr	= f2fs_listxattr,
	.fiemap		= f2fs_fiemap,
	.fileattr_get	= f2fs_fileattr_get,
	.fileattr_set	= f2fs_fileattr_set,
};
```

And the corresponding file-level operations -- a regular file's read/write
path:

```c file=fs/f2fs/file.c lines=5213-5232
const struct file_operations f2fs_file_operations = {
	.llseek		= f2fs_llseek,
	.read_iter	= f2fs_file_read_iter,
	.write_iter	= f2fs_file_write_iter,
	.iopoll		= iocb_bio_iopoll,
	.open		= f2fs_file_open,
	.release	= f2fs_release_file,
	.mmap		= f2fs_file_mmap,
	.flush		= f2fs_file_flush,
	.fsync		= f2fs_sync_file,
	.fallocate	= f2fs_fallocate,
	.unlocked_ioctl	= f2fs_ioctl,
#ifdef CONFIG_COMPAT
	.compat_ioctl	= f2fs_compat_ioctl,
#endif
	.splice_read	= f2fs_file_splice_read,
	.splice_write	= iter_file_splice_write,
	.fadvise	= f2fs_file_fadvise,
	.fop_flags	= FOP_BUFFER_RASYNC,
};
```

-- versus a directory's, which is much smaller (a directory isn't read()
or write()-able the way a regular file is; it's iterated):

```c file=fs/f2fs/dir.c lines=1093-1102
const struct file_operations f2fs_dir_operations = {
	.llseek		= generic_file_llseek,
	.read		= generic_read_dir,
	.iterate_shared	= f2fs_readdir,
	.fsync		= f2fs_sync_file,
	.unlocked_ioctl	= f2fs_ioctl,
#ifdef CONFIG_COMPAT
	.compat_ioctl   = f2fs_compat_ioctl,
#endif
};
```

Down to the Block Layer
========================

None of the operations above touch a storage device directly. When F2FS
actually needs to read or write blocks, it builds a ``struct bio`` (a
block I/O request) and hands it to the kernel's block layer via
``submit_bio()`` -- the same block-layer entry point every other
filesystem and block-backed subsystem uses:

```c file=fs/f2fs/data.c lines=505-513
void f2fs_submit_read_bio(struct f2fs_sb_info *sbi, struct bio *bio,
				 enum page_type type)
{
	WARN_ON_ONCE(!is_read_io(bio_op(bio)));
	trace_f2fs_submit_read_bio(sbi->sb, type, bio);

	iostat_update_submit_ctx(bio, type);
	submit_bio(bio);
}
```

The block layer's own multi-queue design (blk-mq) exists because a single
request queue with a single lock, tuned for mechanical hard disks, became
the bottleneck once fast, highly parallel flash devices (the class of
device F2FS targets) arrived:

```rst file=Documentation/block/blk-mq.rst lines=15-39
Background
----------

Magnetic hard disks have been the de facto standard from the beginning of the
development of the kernel. The Block IO subsystem aimed to achieve the best
performance possible for those devices with a high penalty when doing random
access, and the bottleneck was the mechanical moving parts, a lot slower than
any layer on the storage stack. One example of such optimization technique
involves ordering read/write requests according to the current position of the
hard disk head.

However, with the development of Solid State Drives and Non-Volatile Memories
without mechanical parts nor random access penalty and capable of performing
high parallel access, the bottleneck of the stack had moved from the storage
device to the operating system. In order to take advantage of the parallelism
in those devices' design, the multi-queue mechanism was introduced.

The former design had a single queue to store block IO requests with a single
lock. That did not scale well in SMP systems due to dirty data in cache and the
bottleneck of having a single lock for multiple processors. This setup also
suffered with congestion when different processes (or the same process, moving
to different CPUs) wanted to perform block IO. Instead of this, the blk-mq API
spawns multiple queues with individual entry points local to the CPU, removing
the need for a lock. A deeper explanation on how this works is covered in the
following section (`Operation`_).
```

This is why F2FS's own on-disk design choices (large sequential writes,
avoiding small random updates -- see the F2FS overview lesson and the NAND
flash hardware lesson) matter as much as they do: F2FS shapes its I/O
pattern to suit the flash device sitting below this same block-mq queueing
layer, not just to suit its own on-disk format.

The Page Cache in Between
==========================

In the common case, F2FS's ``read_iter``/``write_iter`` file operations
(see ``f2fs_file_operations`` above) do not go straight to
``submit_bio()`` for every read and write. Ordinary reads, writes, and
mmaps are mediated by the page cache:

```rst file=Documentation/mm/page_cache.rst lines=4-9
Page Cache
==========

The page cache is the primary way that the user and the rest of the kernel
interact with filesystems.  It can be bypassed (e.g. with O_DIRECT),
but normal reads, writes and mmaps go through the page cache.
```

This is why F2FS's checkpoint and writeback logic (see the checkpoint-
related source chunks already indexed from ``fs/f2fs/checkpoint.c``) has
to explicitly flush dirty page-cache pages down to storage at the right
moments -- the page cache is what makes most F2FS reads/writes fast (DRAM
speed, not flash speed), but it also means "written" from an application's
point of view and "physically on flash" are two different moments in
time that F2FS's checkpoint mechanism has to reconcile.

Putting It Together
====================

A single F2FS write, in order, touches: a VFS system call, dispatched
through F2FS's registered ``file_operations``/``inode_operations`` (this
lesson, above); buffered in the page cache until writeback (this lesson,
above); eventually built into a ``bio`` and submitted through the
block-mq layer (this lesson, above); landing on a NAND-flash-backed
device whose erase-before-write constraint is *why* F2FS is a
log-structured filesystem in the first place (see the NAND flash
hardware lesson). No single one of these four layers explains F2FS's
design on its own -- it is the combination that F2FS's design has to
satisfy simultaneously.
