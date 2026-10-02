// Diagnostic dump of SPU / SPURS state, used to investigate SPURS job dispatch stalls
// (e.g. Uncharted 2 "RsxKick: *** Timeout while waiting on RSX SPU kicks").
//
// Triggered from sys_tty_write when the game prints a known stall message. Everything
// here is a racy, read-only snapshot taken from the PPU thread while SPUs keep running.

#include "stdafx.h"
#include "Emu/Memory/vm.h"
#include "Emu/Memory/vm_ptr.h"
#include "Emu/Memory/vm_reservation.h"
#include "Emu/IdManager.h"
#include "Emu/Cell/SPUThread.h"
#include "Emu/Cell/PPUThread.h"
#include "Emu/Cell/lv2/sys_spu.h"
#include "Emu/Cell/Modules/cellSpurs.h"
#include "Emu/Cell/timers.hpp"
#include "Emu/Cell/lv2/sys_rsx.h"
#include "Emu/RSX/RSXThread.h"
#include "Emu/System.h"
#include "Utilities/File.h"

#include <set>
#include <array>
#include <map>
#include <thread>
#include <chrono>

LOG_CHANNEL(spurs_dbg, "SPURSDBG");

// Address of CellSpurs::wklFlag.flag of the first SPURS instance seen (0 = not yet known)
atomic_t<u32> g_spurs_dbg_flag_addr{0};
// 128-byte line containing it, read by JIT code (u32 for a cheap compare)
atomic_t<u32> g_spurs_dbg_flag_line{umax};

namespace
{
	constexpr u32 invalid_spurs = 0u - 0x80u;

	enum dbg_kind : u32
	{
		dbg_spu_putllc = 1, // v0 = flag in reservation snapshot, v1 = flag written, v2 = flag in memory just before
		dbg_spu_putlluc,    // v1 = flag written, v2 = flag in memory just before
		dbg_spu_dma_put,    // v1 = flag written, v2 = flag in memory just before
		dbg_rsx_nv0039,     // v1 = flag after the copy
	};

	struct dbg_event
	{
		u64 time;
		u32 kind;
		u32 who;
		u32 pc;
		u32 v0;
		u32 v1;
		u32 v2;
	};

	std::array<dbg_event, 1024> g_ring{};
	atomic_t<u32> g_ring_pos{0};

	constexpr const char* kind_name(u32 kind)
	{
		switch (kind)
		{
		case dbg_spu_putllc: return "SPU PUTLLC ";
		case dbg_spu_putlluc: return "SPU PUTLLUC";
		case dbg_spu_dma_put: return "SPU DMA PUT";
		case dbg_rsx_nv0039: return "RSX NV0039 ";
		default: return "?";
		}
	}

	u32 read_flag(u32 addr)
	{
		return vm::check_addr(addr) ? +vm::_ref<atomic_be_t<u32>>(addr).load() : 0xdeadbeef;
	}

	u64 res_time(u32 addr)
	{
		return vm::reservation_acquire(addr).load();
	}

	template <typename T>
	bool readable(u32 addr, u32 size = sizeof(T))
	{
		return addr && vm::check_addr(addr, vm::page_readable, size);
	}

	void dump_spurs(std::string& out, u32 spurs_addr)
	{
		if (!readable<CellSpurs>(spurs_addr))
		{
			fmt::append(out, "  CellSpurs 0x%x: not readable\n", spurs_addr);
			return;
		}

		const auto& sp = vm::_ref<CellSpurs>(spurs_addr);

		fmt::append(out, "  CellSpurs 0x%x: line0 rtime=0x%x line1 rtime=0x%x\n", spurs_addr, res_time(spurs_addr), res_time(spurs_addr + 0x80));
		fmt::append(out, "    spuIdling=0x%02x flags1=0x%02x nSpus=%u wklEnabled=0x%08x wklMskB=0x%08x wklSignal1=0x%04x wklSignal2=0x%04x\n",
			sp.spuIdling, sp.flags1, sp.nSpus, +sp.wklEnabled.load(), +sp.wklMskB.load(), +sp.wklSignal1.load(), +sp.wklSignal2.load());
		fmt::append(out, "    wklFlag=0x%08x wklFlagReceiver=0x%02x sysSrvMessage=0x%02x sysSrvOnSpu=0x%02x sysSrvMsgUpdateWorkload=0x%02x\n",
			+sp.wklFlag.flag.load(), +sp.wklFlagReceiver.load(), +sp.sysSrvMessage.load(), sp.sysSrvOnSpu, +sp.sysSrvMsgUpdateWorkload.load());

		const bool wkl32 = (sp.flags1 & SF1_32_WORKLOADS) != 0;

		for (u32 wid = 0; wid < (wkl32 ? 32u : 16u); wid++)
		{
			const u32 i = wid & 0xf;
			const bool hi = wid >= 16;
			const auto& info = hi ? sp.wklInfo2[i] : sp.wklInfo1[i];
			const u8 state = static_cast<u8>(hi ? sp.wklState2[i].load() : sp.wklState1[i].load());

			if (!state && !info.addr)
			{
				continue;
			}

			const u8 ready = hi ? +sp.wklIdleSpuCountOrReadyCount2[i].load() : +sp.wklReadyCount1[i].load();
			const u8 idle_or_r2 = hi ? 0 : +sp.wklIdleSpuCountOrReadyCount2[i].load();
			const u8 cur = wkl32 ? (hi ? sp.wklCurrentContention[i] >> 4 : sp.wklCurrentContention[i] & 0xf) : sp.wklCurrentContention[i];
			const u8 pend = wkl32 ? (hi ? sp.wklPendingContention[i] >> 4 : sp.wklPendingContention[i] & 0xf) : sp.wklPendingContention[i];
			const u8 maxc = wkl32 ? (hi ? sp.wklMaxContention[i].load() >> 4 : sp.wklMaxContention[i].load() & 0xf) : +sp.wklMaxContention[i].load();
			const u8 status = hi ? sp.wklStatus2[i] : sp.wklStatus1[i];
			const u8 event = hi ? +sp.wklEvent2[i].load() : +sp.wklEvent1[i].load();

			fmt::append(out, "    wid %2u: state=%u status=0x%02x event=0x%02x ready=%u idle/ready2=%u cur=%u pend=%u min=%u max=%u pm=0x%x arg=0x%llx size=0x%x uid=%u prio=%016llx\n",
				wid, state, status, event, ready, idle_or_r2, cur, pend, wkl32 ? 0 : sp.wklMinContention[i], maxc,
				info.addr.addr(), +info.arg, +info.size, +info.uniqueId.load(), info.prio64.load());
		}
	}
}

static void spurs_debug_watchdog_start();

void spurs_debug_set_spurs(u32 spurs_addr)
{
	if (g_spurs_dbg_flag_addr.compare_and_swap_test(0, spurs_addr + 0x6c))
	{
		g_spurs_dbg_flag_line = (spurs_addr + 0x6c) & -128;
	}

	spurs_debug_watchdog_start();
}

void spurs_debug_record(u32 kind, u32 who, u32 pc, u32 v0, u32 v1, u32 v2)
{
	const u32 pos = g_ring_pos++ % g_ring.size();
	g_ring[pos] = dbg_event{get_system_time(), kind, who, pc, v0, v1, v2};
}

// Writer hooks: [addr, addr + len) is about to be written from `src` (host pointer to the new bytes)
void spurs_debug_on_write(u32 kind, u32 who, u32 pc, u32 addr, u32 len, const void* src, u32 snapshot_flag)
{
	const u32 flag = g_spurs_dbg_flag_addr;

	if (!flag || flag - addr >= len || flag + 4 - addr > len)
	{
		return;
	}

	be_t<u32> new_val{};
	std::memcpy(&new_val, static_cast<const u8*>(src) + (flag - addr), 4);
	spurs_debug_record(kind, who, pc, snapshot_flag, new_val, read_flag(flag));
}

void spurs_debug_on_rsx_write(u32 addr, u32 len)
{
	const u32 flag = g_spurs_dbg_flag_addr;

	if (flag && flag - addr < len)
	{
		spurs_debug_record(dbg_rsx_nv0039, 0, 0, 0, read_flag(flag), 0);
	}
}

// Called from the SPU LLVM inline DMA path when a PUT targets the flag line
void spurs_debug_on_inline_put(spu_thread* spu, u32 eal, u32 lsa, u32 size)
{
	spurs_debug_on_write(dbg_spu_dma_put, spu->index, spu->pc, eal, size, spu->ls + (lsa & 0x3ffff), 0);
}

void spurs_debug_dump(std::string_view reason)
{
	std::string out;
	fmt::append(out, "==== SPURS debug dump: %s ====\n", reason);

	std::set<u32> spurs_set;

	idm::select<named_thread<spu_thread>>([&](u32 /*id*/, spu_thread& spu)
	{
		const auto ev = spu.ch_events.load();
		const u32 raddr = spu.raddr;

		fmt::append(out, "  SPU[%s] idx=%u pc=0x%05x raddr=0x%x rtime=0x%x cur_rtime=0x%x ev=0x%04x mask=0x%04x waiting=%u spurs=0x%x\n",
			spu.get_name(), spu.index, spu.pc, raddr, spu.rtime, raddr ? res_time(raddr) : 0,
			+ev.events, +ev.mask, +ev.waiting, spu.spurs_addr);

		if (spu.spurs_addr && spu.spurs_addr != invalid_spurs)
		{
			spurs_set.insert(spu.spurs_addr);
		}
	});

	for (u32 addr : spurs_set)
	{
		dump_spurs(out, addr);
	}

	if (const u32 flag = g_spurs_dbg_flag_addr)
	{
		const u64 now = get_system_time();
		const u32 end = g_ring_pos;
		const u32 count = std::min<u32>(end, 96);

		fmt::append(out, "  wklFlag 0x%x = 0x%08x; last %u flag events (t = ms before dump):\n", flag, read_flag(flag), count);

		for (u32 i = end - count; i != end; i++)
		{
			const auto& e = g_ring[i % g_ring.size()];
			fmt::append(out, "    %9.3f %s who=%u pc=0x%05x snapshot=0x%08x written=0x%08x mem_before=0x%08x%s\n",
				(now - e.time) / 1000.0, kind_name(e.kind), e.who, e.pc, e.v0, e.v1, e.v2,
				(e.kind == dbg_spu_putllc && e.v0 != e.v2) ? "  <-- snapshot differs from memory (lost update?)" : "");
		}
	}

	spurs_dbg.error("\n%s", out);
}

// Hang watchdog: when the game stops flipping for a few seconds while emulation is running,
// dump what every guest thread is doing (PPU cia/lr, SPU pc, cpu flags; a few samples each to
// show spin loops) plus the RSX FIFO position and the SPURS state. Fires once per stall.
namespace
{
	void sample_threads(std::string& out)
	{
		struct samples
		{
			std::string name;
			std::string state;
			std::map<u32, u32> pcs; // pc -> hits
			u32 lr = 0;
		};

		std::map<u32, samples> ppus, spus;

		for (int i = 0; i < 20; i++)
		{
			idm::select<named_thread<ppu_thread>>([&](u32 id, ppu_thread& ppu)
			{
				auto& s = ppus[id];
				if (s.name.empty()) s.name = ppu.get_name();
				s.pcs[ppu.cia]++;
				s.lr = static_cast<u32>(ppu.lr);
				s.state = fmt::format("%s", ppu.state.load());
			});

			idm::select<named_thread<spu_thread>>([&](u32 id, spu_thread& spu)
			{
				auto& s = spus[id];
				if (s.name.empty()) s.name = spu.get_name();
				s.pcs[spu.pc]++;
				s.state = fmt::format("%s", spu.state.load());
			});

			std::this_thread::sleep_for(std::chrono::milliseconds(5));
		}

		const auto print = [&](const char* kind, std::map<u32, samples>& m)
		{
			for (auto& [id, s] : m)
			{
				fmt::append(out, "  %s 0x%07x %s lr=0x%x state=[%s] pc:", kind, id, s.name, s.lr, s.state);

				for (auto& [pc, n] : s.pcs)
				{
					fmt::append(out, " 0x%x(%u)", pc, n);
				}

				out += '\n';
			}
		};

		print("PPU", ppus);
		print("SPU", spus);
	}

	// Registers (and, for SPUs, the whole local store) of guest threads that are running rather than waiting.
	// A thread spinning in JIT code is what the hang watchdog is looking for.
	void dump_running_threads(std::string& out)
	{
		idm::select<named_thread<ppu_thread>>([&](u32 id, ppu_thread& ppu)
		{
			if (ppu.state & cpu_flag::wait)
			{
				return;
			}

			fmt::append(out, "  Running PPU 0x%07x %s cia=0x%x lr=0x%x ctr=0x%x (gprs may be stale inside JIT code)\n   ", id, ppu.get_name(), ppu.cia, ppu.lr, ppu.ctr);

			for (u32 i = 0; i < 32; i++)
			{
				fmt::append(out, " r%u=%x", i, ppu.gpr[i]);
			}

			out += '\n';

			// Uncharted 2 main thread spin loop at 0x381478: waits until (*(*(*(r2 - 31164) - 32768)) >> 16) == 0
			if (ppu.cia - 0x381468 < 0x40)
			{
				const u32 toc = static_cast<u32>(ppu.gpr[2]);
				const u32 p1 = vm::check_addr(toc - 31164) ? +vm::_ref<be_t<u32>>(toc - 31164) : 0;
				const u32 p2 = p1 && vm::check_addr(p1 - 32768) ? +vm::_ref<be_t<u32>>(p1 - 32768) : 0;
				const u32 v = p2 && vm::check_addr(p2) ? +vm::_ref<be_t<u32>>(p2) : 0;
				fmt::append(out, "    UC2 spin: toc=0x%x p1=0x%x lock=0x%x value=0x%08x (waits for value>>16 == 0)\n", toc, p1, p2, v);
			}
		});

		idm::select<named_thread<spu_thread>>([&](u32 /*id*/, spu_thread& spu)
		{
			if (spu.state & cpu_flag::wait)
			{
				return;
			}

			fmt::append(out, "  Running SPU %s idx=%u pc=0x%05x srr0=0x%x ch_tag_mask=0x%x mfc_size=%u\n", spu.get_name(), spu.index, spu.pc, spu.srr0, spu.ch_tag_mask, spu.mfc_size);

			for (u32 i = 0; i < 128; i++)
			{
				fmt::append(out, "%s r%-3u=%08x%08x%08x%08x", i % 4 ? "" : "\n   ", i, spu.gpr[i]._u32[3], spu.gpr[i]._u32[2], spu.gpr[i]._u32[1], spu.gpr[i]._u32[0]);
			}

			out += '\n';

			const std::string path = fs::get_cache_dir() + fmt::format("spu%u_ls_%05x.bin", spu.index, spu.pc);

			if (fs::file f{path, fs::rewrite})
			{
				f.write(spu.ls, SPU_LS_SIZE);
				fmt::append(out, "    LS saved to %s\n", path);
			}
		});
	}

	void dump_rsx(std::string& out)
	{
		const auto rsx = rsx::get_current_renderer();

		if (!rsx || !rsx->ctrl)
		{
			out += "  RSX: not initialized\n";
			return;
		}

		const u32 get = rsx->ctrl->get;
		const u32 put = rsx->ctrl->put;
		const u32 ref = rsx->ctrl->ref;
		fmt::append(out, "  RSX: flips=%u get=0x%x put=0x%x ref=0x%x state=[%s]\n", rsx->int_flip_index, get, put, ref, rsx->state.load());

		const u32 ea = rsx->iomap_table.get_addr(get);

		if (ea != umax && vm::check_addr(ea & -16, vm::page_readable, 32))
		{
			fmt::append(out, "  RSX cmd @ get (ea 0x%x):", ea);

			for (u32 i = 0; i < 8; i++)
			{
				fmt::append(out, " %08x", +vm::_ref<be_t<u32>>(ea + i * 4));
			}

			out += '\n';
		}
	}

	[[noreturn]] void watchdog_loop()
	{
		u64 last_flips = umax;
		u64 last_change = get_system_time();
		bool fired = false;

		while (true)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(500));

			const auto rsx = rsx::get_current_renderer();

			if (!Emu.IsRunning() || !rsx)
			{
				last_flips = umax;
				last_change = get_system_time();
				fired = false;
				continue;
			}

			const u64 flips = rsx->int_flip_index;
			const u64 now = get_system_time();

			if (flips != last_flips)
			{
				last_flips = flips;
				last_change = now;
				fired = false;
				continue;
			}

			if (fired || now - last_change < 4'000'000)
			{
				continue;
			}

			fired = true;

			std::string out;
			fmt::append(out, "==== Hang watchdog: no RSX flip for %.1f s ====\n", (now - last_change) / 1e6);
			dump_rsx(out);
			sample_threads(out);
			dump_running_threads(out);
			spurs_dbg.error("\n%s", out);

			spurs_debug_dump("hang watchdog");
		}
	}
}

static void spurs_debug_watchdog_start()
{
	static atomic_t<bool> started{false};

	if (!started.exchange(true))
	{
		std::thread(watchdog_loop).detach();
	}
}
