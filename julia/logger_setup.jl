using Logging, LoggingExtras, Dates

function get_scf_loggers(mol_path::String, basis::String)
    safe_mol = splitext(basename(mol_path))[1]
    safe_basis = replace(basis, r"[^a-zA-Z0-9]" => "_")
    ts = Dates.format(now(), "yyyy-mm-dd_HHMM")

    base_name = "HF_$(safe_mol)_$(safe_basis)_$(ts)"
    info_io = open("$(base_name).log", "w")
    debug_io = open("$(base_name).debug", "w")

    function log_to_streams(io, args, add_timestamp)
        t_prefix = add_timestamp ? "[$(Dates.format(now(), "HH:MM:SS"))] " : ""
        msg = args.level == Logging.Info ? args.message : "[$(args.level)]: $(args.message)"
        println(io, t_prefix, msg)

        # Needed to print matrix values in debug not just text
        for (key, val) in args.kwargs
            println(io, "  └ $key: ", val)
        end
        flush(io)
    end

    terminal_log = MinLevelLogger(FormatLogger((io, args) -> log_to_streams(io, args, false), stdout), Logging.Info)
    info_log     = MinLevelLogger(FormatLogger((io, args) -> log_to_streams(io, args, true), info_io), Logging.Info)
    debug_log    = MinLevelLogger(FormatLogger((io, args) -> log_to_streams(io, args, true), debug_io), Logging.Debug)

    return TeeLogger(terminal_log, info_log, debug_log), info_io, debug_io
end
