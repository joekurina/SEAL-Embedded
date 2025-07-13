namespace csl
{

static constexpr int64_t s_pow2mask[65] = {(((int64_t)1) << 0) - 1,
                                           (((int64_t)1) << 1) - 1,
                                           (((int64_t)1) << 2) - 1,
                                           (((int64_t)1) << 3) - 1,
                                           (((int64_t)1) << 4) - 1,
                                           (((int64_t)1) << 5) - 1,
                                           (((int64_t)1) << 6) - 1,
                                           (((int64_t)1) << 7) - 1,
                                           (((int64_t)1) << 8) - 1,
                                           (((int64_t)1) << 9) - 1,
                                           (((int64_t)1) << 10) - 1,
                                           (((int64_t)1) << 11) - 1,
                                           (((int64_t)1) << 12) - 1,
                                           (((int64_t)1) << 13) - 1,
                                           (((int64_t)1) << 14) - 1,
                                           (((int64_t)1) << 15) - 1,
                                           (((int64_t)1) << 16) - 1,
                                           (((int64_t)1) << 17) - 1,
                                           (((int64_t)1) << 18) - 1,
                                           (((int64_t)1) << 19) - 1,
                                           (((int64_t)1) << 20) - 1,
                                           (((int64_t)1) << 21) - 1,
                                           (((int64_t)1) << 22) - 1,
                                           (((int64_t)1) << 23) - 1,
                                           (((int64_t)1) << 24) - 1,
                                           (((int64_t)1) << 25) - 1,
                                           (((int64_t)1) << 26) - 1,
                                           (((int64_t)1) << 27) - 1,
                                           (((int64_t)1) << 28) - 1,
                                           (((int64_t)1) << 29) - 1,
                                           (((int64_t)1) << 30) - 1,
                                           (((int64_t)1) << 31) - 1,
                                           (((int64_t)1) << 32) - 1,
                                           (((int64_t)1) << 33) - 1,
                                           (((int64_t)1) << 34) - 1,
                                           (((int64_t)1) << 35) - 1,
                                           (((int64_t)1) << 36) - 1,
                                           (((int64_t)1) << 37) - 1,
                                           (((int64_t)1) << 38) - 1,
                                           (((int64_t)1) << 39) - 1,
                                           (((int64_t)1) << 40) - 1,
                                           (((int64_t)1) << 41) - 1,
                                           (((int64_t)1) << 42) - 1,
                                           (((int64_t)1) << 43) - 1,
                                           (((int64_t)1) << 44) - 1,
                                           (((int64_t)1) << 45) - 1,
                                           (((int64_t)1) << 46) - 1,
                                           (((int64_t)1) << 47) - 1,
                                           (((int64_t)1) << 48) - 1,
                                           (((int64_t)1) << 49) - 1,
                                           (((int64_t)1) << 50) - 1,
                                           (((int64_t)1) << 51) - 1,
                                           (((int64_t)1) << 52) - 1,
                                           (((int64_t)1) << 53) - 1,
                                           (((int64_t)1) << 54) - 1,
                                           (((int64_t)1) << 55) - 1,
                                           (((int64_t)1) << 56) - 1,
                                           (((int64_t)1) << 57) - 1,
                                           (((int64_t)1) << 58) - 1,
                                           (((int64_t)1) << 59) - 1,
                                           (((int64_t)1) << 60) - 1,
                                           (((int64_t)1) << 61) - 1,
                                           (((int64_t)1) << 62) - 1,
                                           (int64_t)0x7FFFFFFFFFFFFFFFLL,
                                           ~0};

#ifdef CSL_DEBUG_DUMP_ENABLED
inline void dump_file::header_writer::write(int64_t data)
{
    write(&data, sizeof(int64_t));
}

inline void dump_file::header_writer::write(size_t data)
{
    write(&data, sizeof(size_t));
}

inline void dump_file::header_writer::write(int data)
{
    write(&data, sizeof(int));
}

inline void dump_file::header_writer::write(const void* data, size_t length)
{
    if (m_data.size() < m_pos + length)
    {
        // avoid including <algorithm> just for a single std::max use.
        size_t a = m_pos + length;
        size_t b = m_data.size() * 2;
        size_t larger = a > b ? a : b;
        m_data.resize(larger);
    }
    std::memcpy(m_data.data() + m_pos, data, length);
    m_pos += length;
}

inline void dump_file::header_writer::write_string(const std::string& data)
{
    write(data.c_str(), data.size());

    const uint8_t nullt = 0;
    write(&nullt, 1);
}

inline const size_t dump_file::header_writer::get_size() const
{
    return m_pos;
}

inline const uint8_t* dump_file::header_writer::get_data() const
{
    return m_data.data();
}

inline dump_file::dump_file() {}

inline dump_file::~dump_file()
{
    close();
}

inline void dump_file::open(const std::string& file)
{
    m_file_name = file;
}

inline void dump_file::close()
{
    if (m_file_name == "")
    {
        return;
    }

    header_writer header;
    header.write(ID_DUMP);
    header.write(ID_ROWS);

    size_t row_count = m_row_offsets.size();
    if (m_row_offsets.size() != m_row_names.size())
    {
        csl::error("Number of row offsets did not match number of row names in dump file");
    }
    header.write(row_count);
    for (size_t i = 0; i < row_count; ++i)
    {
        header.write(m_row_offsets[i]);
    }
    for (size_t i = 0; i < row_count; ++i)
    {
        header.write_string(m_row_names[i]);
    }

    header.write(ID_LOCS);
    header.write(m_location_info.size());
    for (auto& loc : m_location_info)
    {
        header.write(loc.first);
        header.write(loc.second.m_bit_width);
        header.write_string(loc.second.m_name);
    }

    std::ofstream f(m_file_name, std::ofstream::binary);
    f.write((const char*)header.get_data(), header.get_size());
    f.write((const char*)m_row_data.get_data(), m_row_data.get_size());
}

inline void dump_file::set_location_info(size_t location, int bit_width, const char* name)
{
    m_location_info[location] = {name, bit_width};
}

inline void dump_file::set_current_cycle(int64_t value)
{
    m_current_cycle = value;
}

inline void dump_file::add_row(int64_t* native_data, size_t native_count, mp_int* wide_data, size_t wide_count, const char* step_name)
{
    if (m_file_name == "")
    {
        return;
    }

    m_row_offsets.push_back(m_row_data.get_size());
    m_row_names.push_back(step_name);

    m_row_data.write(ID_CYCL);
    m_row_data.write(m_current_cycle);

    m_row_data.write(ID_SEGS);
    m_row_data.write(size_t(0));

    m_row_data.write(ID_STEP);
    m_row_data.write(native_count + wide_count);
    m_row_data.write(ID_DATA);
    for (size_t i = 0; i < native_count; ++i)
    {
        m_row_data.write_string(std::to_string(native_data[i]));
    }
#ifdef CSL_USE_GMP
    for (size_t i = 0; i < wide_count; ++i)
    {
        char wide[2048];
        wide_data[i].str(wide, 2048);
        m_row_data.write_string(wide);
    }
#endif
}

inline void dump_file::add_row(int64_t* native_data, size_t native_count, mp_int* wide_data, size_t wide_count, const char* step_name,
                               int64_t* seg_cycles, size_t seg_cycle_count, int64_t update_cycle)
{
    if (m_file_name == "")
    {
        return;
    }

    m_row_offsets.push_back(m_row_data.get_size());
    m_row_names.push_back(step_name);

    m_row_data.write(ID_CYCL);
    m_row_data.write(m_current_cycle);

    m_row_data.write(ID_SEGS);
    m_row_data.write(seg_cycle_count + 1);
    for (size_t i = 0; i < seg_cycle_count; ++i)
    {
        m_row_data.write(seg_cycles[i]);
    }
    m_row_data.write(update_cycle);

    m_row_data.write(ID_STEP);
    m_row_data.write(native_count + wide_count);
    m_row_data.write(ID_DATA);
    for (size_t i = 0; i < native_count; ++i)
    {
        m_row_data.write_string(std::to_string(native_data[i]));
    }
#ifdef CSL_USE_GMP
    for (size_t i = 0; i < wide_count; ++i)
    {
        char wide[2048];
        wide_data[i].str(wide, 2048);
        m_row_data.write_string(wide);
    }
#endif
}

#endif // CSL_DEBUG_DUMP_ENABLED

inline std::string to_binary(uint64_t a, unsigned width)
{
    std::string z(width, '\0');
    uint64_t pow2 = ((uint64_t)1) << (width - 1);
    for (unsigned i = 0; i < width; i++)
    {
        if (i > 0)
        {
            pow2 = pow2 >> 1;
        }
        if (a >= pow2)
        {
            z[i] = '1';
            a -= pow2;
        }
        else
        {
            z[i] = '0';
        }
    }
    return z;
}

inline bool get_raw_stm_line(std::ifstream& file, std::vector<std::string>& values)
{
    values.clear();

    std::string line;
    if (std::getline(file, line))
    {
        std::istringstream iss(line);
        while (!iss.eof())
        {
            std::string value;
            iss >> value;
            if (value.size() > 0)
            {
                values.push_back(value);
            }
        }
        return values.size() > 0;
    }
    return false;
}

inline bool compare_stm_files(const char* result_file_name, const char* reference_file_name)
{
    std::vector<std::string> ref_line, result_line;

    std::ifstream result_file(result_file_name);
    std::ifstream reference_file(reference_file_name);
    std::streamoff result_lines = std::count(std::istreambuf_iterator<char>(result_file), std::istreambuf_iterator<char>(), '\n');
    std::streamoff ref_lines = std::count(std::istreambuf_iterator<char>(reference_file), std::istreambuf_iterator<char>(), '\n');
    if (result_lines != ref_lines)
    {
        std::string err = format("Line count mismatch between reference (%d) and result files (%d)", ref_lines, result_lines);
        error(err.c_str());
        return false;
    }

    result_file.clear();
    result_file.seekg(0);
    reference_file.clear();
    reference_file.seekg(0);
    size_t line = 0;
    while (get_raw_stm_line(reference_file, ref_line) && get_raw_stm_line(result_file, result_line))
    {
        if (ref_line.size() != result_line.size())
        {
            error("Stimulus column count mismatch between reference and result files");
            return false;
        }
        for (size_t i = 0; i < ref_line.size(); ++i)
        {
            if (ref_line[i] != result_line[i])
            {
                std::string err = format("Mismatch on result line %d column %d. Expected <%s> but got <%s>.", line, i,
                                         ref_line[i].c_str(), result_line[i].c_str());
                error(err.c_str());
                return false;
            }
        }
        ref_line.clear();
        result_line.clear();
        ++line;
    }
    return true;
}

inline bool stimulus_file::get_u32(uint32_t& value)
{
    if (m_stm_file.is_open() && (m_line_position < m_current_line.size()))
    {
        char* last;
        int32_t int_value = strtol(m_current_line[m_line_position++].c_str(), &last, 10);
        std::memcpy(&value, &int_value, sizeof(value));
        return true;
    }
    return false;
}

inline void stimulus_file::reset_file()
{
    m_stm_file.clear();
    m_stm_file.seekg(0, std::ios::beg);
}

inline void stimulus_file::discover_format()
{
    if (m_stm_format != StimulusFormat::UNKNOWN)
    {
        return;
    }

    if (m_stm_file.eof())
    {
        m_stm_format = StimulusFormat::SIGNED;
        return;
    }

    std::string line;
    while (std::getline(m_stm_file, line))
    {
        std::stringstream linesstream(line);
        std::string stringValue;
        // read in integer tokens from the stream
        while (linesstream >> stringValue)
        {
            if (stringValue.length() == 0)
            {
                continue;
            }

            // binary numbers are padded with 0s
            if ((stringValue.length() > 1) && (stringValue[0] == '0'))
            {
                m_stm_format = StimulusFormat::BINARY;
                reset_file();
                return;
            }

            if (stringValue.find_first_of("-23456789") != std::string::npos)
            {
                m_stm_format = StimulusFormat::SIGNED;
                reset_file();
                return;
            }
        }
    }
}

inline stimulus_file::stimulus_file(const std::string& fileName, StimulusFormat format) : m_line_position(0), m_stm_format(format)
{
    open(fileName, format);
}

inline stimulus_file::stimulus_file() : m_line_position(0), m_stm_format(StimulusFormat::UNKNOWN)
{
    m_line_position = 0;
}

inline stimulus_file::~stimulus_file()
{
    // close the file if it has been left open
    close();
}

inline void stimulus_file::close()
{
    if (is_open())
    {
        m_stm_file.close();
    }
}

inline bool stimulus_file::open(const std::string& fileName, StimulusFormat format)
{
    if (is_open())
    {
        close();
    }

    m_stm_format = format;
    m_stm_file.open(fileName.c_str(), std::ios::in);
    bool open = is_open();
    if (!open)
    {
        fmt_warning("Could not open stimulus file (%s).\n", fileName.c_str());
    }
    else
    {
        discover_format();
    }
    return open;
}

inline bool stimulus_file::is_open() const
{
    return m_stm_file.is_open();
}

inline bool stimulus_file::is_line_fully_read() const
{
    return m_line_position == m_current_line.size();
}

inline bool stimulus_file::next_line()
{
    if (m_stm_file.eof())
    {
        return false;
    }

    if (!is_line_fully_read())
    {
        warning("Previous line was not fully read");
    }

    // clear current line
    m_line_position = 0;
    m_current_line.clear();

    // read a line into the vector
    std::string line;
    if (std::getline(m_stm_file, line))
    {
        std::stringstream linesstream(line);

        std::string value;
        // read in string tokens from the stream
        while (linesstream >> value)
        {
            m_current_line.push_back(value);
        }

        return true;
    }

    return false;
}

inline bool stimulus_file::skip_lines(size_t n)
{
    bool success = true;
    for (size_t i = 0; i < n; ++i)
    {
        success &= skip_line();
    }
    return success;
}

inline bool stimulus_file::skip_line()
{
    if (m_stm_file.eof())
    {
        return false;
    }

    if (!is_line_fully_read())
    {
        warning("Previous line was not fully read");
    }

    // clear current line
    m_line_position = 0;
    m_current_line.clear();

    // read a line into the vector
    std::string line;
    if (std::getline(m_stm_file, line))
    {
        return true;
    }

    return false;
}

inline void stimulus_file::skip(int count)
{
    int64_t dummy;
    for (int i = 0; i < count; ++i)
    {
        get(dummy, 32);
    }
}

#ifdef CSL_USE_GMP
inline bool stimulus_file::get(uint32_t* values, size_t bit_width, size_t capacity)
{
    for (size_t i = 0; i < capacity; ++i)
    {
        values[i] = 0;
    }

    if (bit_width <= 0)
    {
        return false;
    }

    if (m_stm_format == StimulusFormat::SIGNED)
    {
        for (size_t i = 0; i < capacity; ++i)
        {
            if (!get_u32(values[i]))
            {
                return false;
            }
        }
    }
    else
    {
        error("Non-signed stimulus files currently unsupported for wrapper ATBs");
    }

    return false;
}

inline bool stimulus_file::get(mp_int& var, size_t bit_width)
{
    if (bit_width <= 0)
    {
        var = 0;
        return false;
    }

    var = 0;
    if (m_stm_format == StimulusFormat::SIGNED)
    {
        // as per get_int_arb but no processing on the ints required
        int32_t num_ints = static_cast<int32_t>(((bit_width - 1) / 32) + 1);
        std::vector<uint32_t> values(num_ints);
        for (int32_t i = 0; i < num_ints; ++i)
        {
            if (!get_u32(values[i]))
            {
                var = 0;
                return false;
            }
        }
        var.set_from_uint_array(values.data(), values.size(), bit_width, m_temps);
        return true;
    }
    else
    {
        if (m_stm_file.is_open() && (m_line_position < m_current_line.size()))
        {
            var = m_current_line[m_line_position++].c_str();
            return true;
        }
    }

    return false;
}
#endif

inline bool stimulus_file::get(int64_t& value, size_t bit_width)
{
    if (bit_width <= 0)
    {
        value = 0ull;
        return false;
    }

    uint64_t result = 0;
    if (m_stm_file.is_open() && (m_line_position < m_current_line.size()))
    {
        if (m_stm_format == StimulusFormat::SIGNED)
        {
            uint32_t result_u32;
            get_u32(result_u32);
            result = result_u32;

            size_t count = 1 + ((bit_width - 1) / 32);
            if (count > 1)
            {
                if (m_line_position >= m_current_line.size())
                {
                    return false;
                }
                uint32_t v = 0;
                get_u32(v);
                result |= static_cast<uint64_t>(v) << 32;
            }

            std::memcpy(&value, &result, sizeof(value));
            return true;
        }
        else if (m_stm_format == StimulusFormat::BINARY)
        {
            if (m_stm_file.is_open() && (m_line_position < m_current_line.size()))
            {
                char* last;
                int64_t int_value = strtoll(m_current_line[m_line_position++].c_str(), &last, 2);
                std::memcpy(&value, &int_value, sizeof(value));
                return true;
            }
            return false;
        }
    }
    value = 0ull;
    return false;
}

inline bool stimulus_file::get(int32_t& value, size_t bit_width)
{
    int64_t int_value = 0;
    bool result = get(int_value, bit_width);
    std::memcpy(&value, &int_value, sizeof(int32_t));
    return result;
}

inline bool stimulus_file::get(int16_t& value, size_t bit_width)
{
    int64_t int_value = 0;
    bool result = get(int_value, bit_width);
    std::memcpy(&value, &int_value, sizeof(int16_t));
    return result;
}

inline bool stimulus_file::get(int8_t& value, size_t bit_width)
{
    int64_t int_value = 0;
    bool result = get(int_value, bit_width);
    std::memcpy(&value, &int_value, sizeof(int8_t));
    return result;
}

inline bool stimulus_file::get(double& value)
{
    int64_t int_value = 0;
    bool result = get(int_value, 64);
    std::memcpy(&value, &int_value, sizeof(double));
    return result;
}

inline bool stimulus_file::get(float& value)
{
    int64_t int_value = 0;
    bool result = get(int_value, 32);
    std::memcpy(&value, &int_value, sizeof(float));
    return result;
}

inline output_stimulus_file::output_stimulus_file(const char* filename)
{
    open(filename);
}

inline output_stimulus_file::~output_stimulus_file()
{
    close();
}

inline void output_stimulus_file::open(const char* filename)
{
    close();
    m_file.open(filename);
    if (!m_file.is_open())
    {
        fmt_error("Could not open file: %s", filename);
    }
}

inline void output_stimulus_file::close()
{
    if (m_file.is_open())
    {
        m_file.close();
    }
}

inline void output_stimulus_file::write_stm_data_impl(int64_t v, int64_t width)
{
    if (!m_file.is_open())
    {
        fmt_error("Attempting to write stimulus but no file was opened.");
    }

#ifdef CSL_BINARY_STM_OUT
    x &= bit_mask_i64(width);
    m_file << toBinary<width>(x) << " ";
#elif CSL_UNSPLIT_STM_OUT
    x &= bit_mask_i64(width);
    m_file << x << " ";
#else
    char buffer[32];
    for (int bit = 0; bit < width; bit += 32)
    {
        if (bit + 32 > width)
        {
            v &= s_pow2mask[width - bit];
        }
        int r = std::snprintf(buffer, 32, "%d ", (int)(v & 0xffffffff));
        m_file.write(buffer, r);
        v >>= 32;
    }
#endif
}

inline void output_stimulus_file::write_stm_data(int64_t v, int64_t width)
{
    int64_t x = 0;
    std::memcpy(&x, &v, sizeof(int64_t));
    write_stm_data_impl(x, width);
}

inline void output_stimulus_file::write_stm_data(int32_t v, int64_t width)
{
    int64_t x = 0;
    std::memcpy(&x, &v, sizeof(int32_t));
    write_stm_data_impl(x, width);
}

inline void output_stimulus_file::write_stm_data(int16_t v, int64_t width)
{
    int64_t x = 0;
    std::memcpy(&x, &v, sizeof(int16_t));
    write_stm_data_impl(x, width);
}

inline void output_stimulus_file::write_stm_data(int8_t v, int64_t width)
{
    int64_t x = 0;
    std::memcpy(&x, &v, sizeof(int8_t));
    write_stm_data_impl(x, width);
}

inline void output_stimulus_file::write_stm_data(double v)
{
    int64_t x = 0;
    std::memcpy(&x, &v, sizeof(double));
    write_stm_data_impl(x, 64);
}

inline void output_stimulus_file::write_stm_data(float v)
{
    int64_t x = 0;
    std::memcpy(&x, &v, sizeof(float));
    write_stm_data_impl(x, 32);
}

#ifdef CSL_USE_GMP
inline void output_stimulus_file::write_stm_data(const mp_int& v, int64_t width)
{
    if (!m_file.is_open())
    {
        fmt_error("Attempting to write stimulus but no file was opened.");
    }

    mp_int& masked_val = m_temps.values[0];
    mp_int& curr = m_temps.values[1];
    curr = v;

#ifdef CSL_BINARY_STM_OUT
    (void)width;
    char value[1024];
    curr.str_bin(value, 1024);
    m_file << value << " ";
#elif CSL_UNSPLIT_STM_OUT
    (void)width;
    char value[1024];
    curr.str(value, 1024);
    m_file << value << " ";
#else
    char buffer[32];
    for (int bit = 0; bit < width; bit += 32)
    {
        mpz_and(masked_val.get(), curr.get(), m_mask.get());
        int int_value = (int)static_cast<unsigned int>(mpz_get_ui(masked_val.get()));
        if (width - bit <= 32)
        {
            int_value &= s_pow2mask[width - bit];
        }
        int r = std::snprintf(buffer, 32, "%d ", int_value);
        m_file.write(buffer, r);
        mpz_fdiv_q_2exp(curr.get(), curr.get(), 32);
    }
#endif
}

#endif

inline void output_stimulus_file::next_line()
{
    if (!m_file.is_open())
    {
        fmt_error("Attempting to write stimulus but no file was opened.");
    }
    m_file << "\n";
    ++m_line_count;
}

inline size_t output_stimulus_file::get_line_count() const
{
    return m_line_count;
}

} // namespace csl