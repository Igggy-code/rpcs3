#include "trophy_notification_helper.h"
#include "trophy_notification_frame.h"

#include "../Emu/IdManager.h"
#include "../Emu/emu_callbacks.h"
#include "../Emu/System.h"

#include "../Emu/RSX/Overlays/overlay_manager.h"
#include "../Emu/RSX/Overlays/overlay_trophy_notification.h"

#include "Utilities/File.h"

#include <QPointer>

s32 trophy_notification_helper::ShowTrophyNotification(const SceNpTrophyDetails& trophy, const std::vector<uchar>& trophy_icon_buffer)
{
	if (auto manager = g_fxo->try_get<rsx::overlays::display_manager>())
	{
		// Allow adding more than one trophy notification. The notification class manages scheduling
		auto popup = std::make_shared<rsx::overlays::trophy_notification>();
		return manager->add(popup, false)->show(trophy, trophy_icon_buffer);
	}

	if (!Emu.HasGui())
	{
		return 0;
	}

	// The helper is a temporary owned by the caller and is destroyed before this runs on the main
	// thread: capture the window itself (guarded, it may be closed meanwhile), never 'this'
	Emu.CallFromMainThread([trophy, trophy_icon_buffer, game_window = QPointer<QWindow>(m_game_window)]
	{
		if (!game_window)
		{
			return;
		}

		trophy_notification_frame* trophy_notification = new trophy_notification_frame(trophy_icon_buffer, trophy, game_window->frameGeometry().height() / 10);

		// Move notification to upper lefthand corner
		trophy_notification->move(game_window->mapToGlobal(QPoint(0, 0)));
		trophy_notification->show();

		g_emu_callbacks.play_sound(fs::get_config_dir() + "sounds/snd_trophy.wav", std::nullopt);
	});

	return 0;
}
