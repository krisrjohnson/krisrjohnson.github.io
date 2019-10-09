---
layout: default
title: Jekyll Notes
---

Fantastic [cheatsheet](https://devhints.io/jekyll)

# Setup
Jekyll provides an excellent and fast [Quickstart](https://jekyllrb.com/docs/), plus [Step by Step](https://jekyllrb.com/docs/step-by-step/01-setup/) tutorial.
Basic steps:
1. download ruby w/ devkit (needed for wdm), 
1. run the ridk install in last step of ruby installer, 
1. in cmd.exe run `gemy install jekyll bundler` to get jekyll
1. `bundle install` in the site's repo to download missing gems specified in gemfile
1. `bundle exec jekyll serve` to local host site, which should automatically watch for changes
	1. `bundle exec jekyll build --watch` won't locally host and is used for actual live deploys, github calls this automatically on `git push`

If having issues with jekyll listening during build or serve to local changes, try adding to your gemfile `gem 'wdm', '~> 0.1.1', :install_if => Gem.win_platform?` and rerun `bundle install`

# Liquid Tags
Building block of jekyll's power, allowing mucho flexibilitio. 

Not worth looking up how to properly escape liquid tags while writing markdown so they won't be executed at build. Basic ideas are using double curly brackets around variables and single curly brackets with percent signs around logic statements.

The [basics](https://jekyllrb.com/docs/step-by-step/04-layouts/), example [conditional](https://jekyllrb.com/docs/configuration/environments/), and full Liquid [reference](https://jekyllrb.com/docs/liquid/)

# Templates
Create a `_layouts` folder and add default.html, which is now a template/layout. Using liquid tags, can abstract parts of the page within default.html so when writing default pages such as `index.html`, only can do so in markdown without worrying about the header, nav bar, footer, etc. 

## Ex: footer.html
Create `_include/footer.html`, this can now be referenced via liquid tag in any other file, and is called in `_layouts/default.html` via a logical liquid tag (`include footer.html` surrounded by curly brackets with percent signs). 

The actual `_include/footer.html` file contains a `<footer>` tag surrounding an unordered list populated by liquid code referencing `_data/footer.yml` by calling a `for item in site.data.footer` liquid conditional. The contents of any directory prefaced with "\_" can be called via `site.directory_name`

To edit the footer, say to add a twitter account, you'd open up the YAML file and `-name: twitter \n -link: twitter.com/some_account` and then the footer would additionally display a clickable link of text "twitter".

# Customization
Can add permalink variable to __front matter__ (everything between '---'), default behavior for posts: /year/month/day/title.html because posts are saved as YYYY-MM-DD-title.md

Interesting [article](https://ben.balter.com/2015/02/20/jekyll-collections/) on collections. Use for say organizing people in a site's employees page. Jekyll collections [documentation](https://jekyllrb.com/docs/collections/)


