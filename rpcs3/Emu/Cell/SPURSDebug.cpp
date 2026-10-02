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

#include <set>

LOG_CHANNEL(spurs_dbg, "SPURSDBG");

namespace
{
	constexpr u32 invalid_spurs = 0u - 0x80u;

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

	spurs_dbg.error("\n%s", out);
}
