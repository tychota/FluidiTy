#include "fluidity/base/processor_description_text.h"

#include <gtest/gtest.h>

#include <string>

namespace fluidity {
namespace {

TEST(FindCpuModelInText, NormalX86Text_ReturnsTrimmedName) {
  const std::string text =
      "processor\t: 0\nvendor_id\t: AuthenticAMD\nmodel name\t:  AMD Ryzen 9 7950X 16-Core Processor  \n"
      "model name\t: second line is ignored\n";
  EXPECT_EQ(FindCpuModelInText(text), "AMD Ryzen 9 7950X 16-Core Processor");
}

TEST(FindCpuModelInText, NoModelNameLine_ReturnsEmpty) {
  const std::string text = "processor\t: 0\nBogoMIPS\t: 50.00\nCPU part\t: 0xd0c\n";
  EXPECT_TRUE(FindCpuModelInText(text).empty());
}

TEST(FindCpuModelInText, EmptyText_ReturnsEmpty) {
  EXPECT_TRUE(FindCpuModelInText("").empty());
}

TEST(FindCpuModelInText, EmptyValue_ReturnsEmpty) {
  EXPECT_TRUE(FindCpuModelInText("model name\t:\nprocessor\t: 1\n").empty());
}

TEST(FindCpuModelInText, ValueOfOnlySpaces_ReturnsEmpty) {
  EXPECT_TRUE(FindCpuModelInText("model name\t:   \t \n").empty());
}

TEST(FindCpuModelInText, LastLineWithoutNewline_IsRead) {
  EXPECT_EQ(FindCpuModelInText("processor\t: 0\nmodel name\t: Some CPU"), "Some CPU");
}

}  // namespace
}  // namespace fluidity
