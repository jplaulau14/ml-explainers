from tiling_explainer.conditions import PowerState, parse_power, refusal

SETTINGS = "Currently in use:\n standby              1\n lowpowermode         {}\n"
ON_AC = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t28%; charging; present: true\n"
ON_BATTERY = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t36%; discharging\n"
AC = PowerState(ac_power=True, low_power_mode=False)


def test_pmset_on_ac_with_low_power_mode_off() -> None:
    assert parse_power(SETTINGS.format(0), ON_AC) == AC


def test_pmset_on_battery_with_low_power_mode_on() -> None:
    state = parse_power(SETTINGS.format(1), ON_BATTERY)
    assert state == PowerState(ac_power=False, low_power_mode=True)


def test_publication_refused_on_battery_low_power_mode_or_dirty_tree() -> None:
    assert refusal([AC, AC], dirty=False) is None
    assert refusal([AC, PowerState(False, False)], dirty=False) == "running on battery"
    assert refusal([AC, PowerState(True, True)], dirty=False) == "Low Power Mode is on"
    assert refusal([AC], dirty=True) == "uncommitted changes in tiling-matmul/ outside out/"
