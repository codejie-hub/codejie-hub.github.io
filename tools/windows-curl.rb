# frozen_string_literal: true
# Optional native-library bootstrap for HTMLProofer on Windows.
# Usage: BLOG_CURL_DLL=<Git's libcurl-4.dll> RUBYOPT=-r./tools/windows-curl.rb
# Copies one explicitly named DLL into the ignored project cache, preserving originals.
if RUBY_PLATFORM.match?(/mingw|mswin/) && ENV['BLOG_CURL_DLL']
  require 'fiddle'
  require 'fileutils'
  curl_source = File.expand_path(ENV.fetch('BLOG_CURL_DLL'))
  raise "libcurl not found: #{curl_source}" unless File.file?(curl_source)
  task_cache = File.expand_path('../.qa/native', __dir__)
  FileUtils.mkdir_p(task_cache)
  curl_alias = File.join(task_cache, 'libcurl.dll')
  FileUtils.cp(curl_source, curl_alias) unless File.file?(curl_alias)
  kernel = Fiddle.dlopen('kernel32.dll')
  directory = Fiddle::Function.new(kernel['SetDllDirectoryW'], [Fiddle::TYPE_VOIDP], Fiddle::TYPE_INT)
  wide_path = File.dirname(curl_source).encode('UTF-16LE').b + [0].pack('v')
  raise 'SetDllDirectoryW failed' if directory.call(wide_path).zero?
  # Keep the handle alive for Ethon/FFI's subsequent load by the conventional name.
  BLOG_CURL_HANDLE = Fiddle.dlopen(curl_alias)
end
