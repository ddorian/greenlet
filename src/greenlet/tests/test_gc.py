import gc

import struct
import weakref
import sys
import greenlet


from . import TestCase
from . import RUNNING_ON_FREETHREAD_BUILD
from .leakcheck import fails_leakcheck_on_py314_or_less
# These only work with greenlet gc support
# which is no longer optional.
assert greenlet.GREENLET_USE_GC

class TestGC(TestCase):
    def test_dead_circular_ref(self):
        o = weakref.ref(greenlet.greenlet(greenlet.getcurrent).switch())
        gc.collect()
        if o() is not None:
            print("O IS NOT NONE.", sys.getrefcount(o()))
        self.assertIsNone(o())
        self.assertFalse(gc.garbage, gc.garbage)

    def test_circular_greenlet(self):
        class circular_greenlet(greenlet.greenlet):
            self = None
        o = circular_greenlet()
        o.self = o
        o = weakref.ref(o)
        gc.collect()
        self.assertIsNone(o())
        self.assertFalse(gc.garbage, gc.garbage)

    def test_inactive_ref(self):
        class inactive_greenlet(greenlet.greenlet):
            def __init__(self):
                greenlet.greenlet.__init__(self, run=self.run)

            def run(self):
                pass
        o = inactive_greenlet()
        o = weakref.ref(o)
        gc.collect()
        self.assertIsNone(o())
        self.assertFalse(gc.garbage, gc.garbage)

    @fails_leakcheck_on_py314_or_less
    def test_finalizer_crash(self):
        # This test is designed to crash when active greenlets
        # are made garbage collectable, until the underlying
        # problem is resolved. How does it work:
        # - order of object creation is important
        # - array is created first, so it is moved to unreachable first
        # - we create a cycle between a greenlet and this array
        # - we create an object that participates in gc, is only
        #   referenced by a greenlet, and would corrupt gc lists
        #   on destruction, the easiest is to use an object with
        #   a finalizer
        # - because array is the first object in unreachable it is
        #   cleared first, which causes all references to greenlet
        #   to disappear and causes greenlet to be destroyed, but since
        #   it is still live it causes a switch during gc, which causes
        #   an object with finalizer to be destroyed, which causes stack
        #   corruption and then a crash

        class object_with_finalizer(object):
            def __del__(self):
                pass
        array = []
        parent = greenlet.getcurrent()
        def greenlet_body():
            greenlet.getcurrent().object = object_with_finalizer()
            try:
                parent.switch()
            except greenlet.GreenletExit:
                print("Got greenlet exit!")
            finally:
                del greenlet.getcurrent().object
        g = greenlet.greenlet(greenlet_body)
        g.array = array
        array.append(g)
        g.switch()
        del array
        del g
        greenlet.getcurrent()
        gc.collect()

    def test_issue515_freethread_c_stack_refs(self):
        # Guards issue #515: a new greenlet inherited the parent's C-stack refs
        # and the free-threaded collector segfaulted following the dangling
        # nodes. This has to run out of process because the regression is a hard
        # crash, not something we can catch.
        # https://github.com/python-greenlet/greenlet/issues/515
        if not RUNNING_ON_FREETHREAD_BUILD:
            self.skipTest("Only free-threaded builds are affected")
        output = self.run_script('fail_issue_515_freethread_gc.py')
        self.assertIn('ISSUE 515 OK', output)

    def test_c_stack_refs_suspended_gc(self):
        # Issue #515: a greenlet suspended while holding a _PyCStackRef must have
        # those refs visited by tp_traverse, or the free-threaded collector frees
        # an object reachable only through the suspended C stack. Runs the repro
        # out of process. https://github.com/python-greenlet/greenlet/issues/515
        if not RUNNING_ON_FREETHREAD_BUILD:
            self.skipTest("Only free-threaded builds are affected")
        output = self.run_script('fail_c_stack_refs_suspended_gc.py')
        self.assertIn('C STACK REFS GC OK', output)

    def _c_stack_refs_probe(self):
        if not RUNNING_ON_FREETHREAD_BUILD or sys.version_info < (3, 14):
            self.skipTest("Only free-threaded 3.14+ resolves the offset")
        mod = greenlet._greenlet
        return mod._probe_c_stack_refs_offset, mod._C_STACK_REFS_OFFSET

    def test_c_stack_refs_offset_resolved(self):
        # Issue #527: 3.14.4 appended a PyThreadState field, which moved
        # _PyThreadStateImpl.c_stack_refs by 8 bytes inside a released series. A
        # wheel built against one 3.14.x read the wrong word on another and
        # segfaulted on the first switch, so we ask the interpreter where the
        # field is rather than trusting offsetof().
        _, offset = self._c_stack_refs_probe()
        self.assertGreater(offset, 0)
        self.assertEqual(offset % struct.calcsize('P'), 0)

    def test_c_stack_refs_offset_survives_a_wrong_start(self):
        # The regression this guards: a build whose compile-time offsetof() is
        # off by a pointer or two still finds the real field.
        probe, offset = self._c_stack_refs_probe()
        for bias in (-24, -16, -8, 0, 8, 16, 24):
            self.assertEqual(probe(offset + bias), offset, bias)

    def test_c_stack_refs_offset_admits_defeat(self):
        # Out of range it reports 0 instead of guessing, which is what turns an
        # unrecognized layout into an ImportError rather than a crash.
        probe, offset = self._c_stack_refs_probe()
        self.assertEqual(probe(offset + 200), 0)
        self.assertEqual(probe(0), 0)
        with self.assertRaises(ValueError):
            probe(-1)

    def test_crashing_deferred_object(self):
        if sys.version_info < (3, 15):
            self.skipTest("Test is 3.15+ only")
        import doctest
        def with_doctest():
            """
            >>> import gc
            >>> from greenlet import getcurrent, greenlet, GreenletExit
            >>> def outer():
            ...     gc.collect()
            >>> outer_glet = greenlet(outer)
            >>> outer_glet.switch()
            """
        doctest.run_docstring_examples(with_doctest, dict())

    def test_cycle_in_suspended_frame(self):
        if sys.version_info < (3, 15):
            self.skipTest("Test is 3.15+ only")
        import doctest
        def with_doctest():
            """
            >>> import gc
            >>> from greenlet import getcurrent, greenlet
            >>> class Cycle:
            ...     def __del__(self):
            ...         print("(Running finalizer)")
            >>> def collect_it():
            ...     print("Collecting garbage")
            ...     gc.collect()
            >>> def inner():
            ...     cycle1 = Cycle()
            ...     cycle2 = Cycle()
            ...     cycle1.cycle = cycle2
            ...     cycle2.cycle = cycle1
            ...     getcurrent().parent.switch()
            >>> def outer():
            ...     glet = greenlet(inner)
            ...     glet.switch()
            ...     collect_it()

            >>> outer_glet = greenlet(outer)
            >>> outer_glet.switch()
            Collecting garbage
            >>> outer_glet.dead
            True
            >>> collect_it()
            Collecting garbage
            (Running finalizer)
            (Running finalizer)
            """
        doctest.run_docstring_examples(with_doctest, dict())
